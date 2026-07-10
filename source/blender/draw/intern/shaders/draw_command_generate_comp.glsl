/* SPDX-FileCopyrightText: 2022-2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/**
 * Convert DrawPrototype into draw commands.
 */

#include "draw_view_infos.hh"

COMPUTE_SHADER_CREATE_INFO(draw_command_generate)

#define atomicAddAndGet(dst, val) (atomicAdd(dst, val) + val)

/* WORKAROUND (WebGPU/Tint): the *_counter members are accessed atomically, so
 * every access must be an atomic op and the containing struct must never be
 * loaded wholesale (Tint's SPIR-V atomics lowering asserts on mixed access).
 * Plain reads use atomicAdd(x, 0), resets use atomicExchange. */

/* This is only called by the last thread executed over the group's prototype draws. */
void write_draw_call(DrawGroup group, uint group_id)
{
  const bool indexed_draw = group.base_index != -1;

  const uint back_facing_len = atomicAdd(group_buf[group_id].back_facing_counter, 0u);
  const uint back_facing_start = group.start * uint(view_len);
  const uint front_facing_len = atomicAdd(group_buf[group_id].front_facing_counter, 0u);
  const uint front_facing_start = (group.start + (group.len - group.front_facing_len)) *
                                  uint(view_len);

  /* Back-facing command. */
  DrawCommand cmd;
  if (indexed_draw) {
    DrawCommandIndexed cmd_indexed;
    cmd_indexed.vertex_len = uint(group.vertex_len);
    cmd_indexed.instance_len = back_facing_len;
    cmd_indexed.vertex_first = uint(group.vertex_first);
    cmd_indexed.base_index = uint(group.base_index);
    cmd_indexed.instance_first = back_facing_start;
    cmd.indexed() = cmd_indexed;
  }
  else {
    DrawCommandArray cmd_array;
    cmd_array.vertex_len = uint(group.vertex_len);
    cmd_array.instance_len = back_facing_len;
    cmd_array.vertex_first = uint(group.vertex_first);
    cmd_array.instance_first = back_facing_start;
    cmd.array() = cmd_array;
  }
  command_buf[group_id * 2 + 0] = cmd;

  /* Front-facing command. */
  if (indexed_draw) {
    DrawCommandIndexed cmd_indexed;
    cmd_indexed.vertex_len = uint(group.vertex_len);
    cmd_indexed.instance_len = front_facing_len;
    cmd_indexed.vertex_first = uint(group.vertex_first);
    cmd_indexed.base_index = uint(group.base_index);
    cmd_indexed.instance_first = front_facing_start;
    cmd.indexed() = cmd_indexed;
  }
  else {
    DrawCommandArray cmd_array;
    cmd_array.vertex_len = uint(group.vertex_len);
    cmd_array.instance_len = front_facing_len;
    cmd_array.vertex_first = uint(group.vertex_first);
    cmd_array.instance_first = front_facing_start;
    cmd.array() = cmd_array;
  }
  command_buf[group_id * 2 + 1] = cmd;

  /* Reset the counters for a next command gen dispatch. Avoids re-sending the whole data just
   * for this purpose. Only the last thread will execute this so it is thread-safe. */
  atomicExchange(group_buf[group_id].front_facing_counter, 0u);
  atomicExchange(group_buf[group_id].back_facing_counter, 0u);
  atomicExchange(group_buf[group_id].total_counter, 0u);
}

void main()
{
  int proto_id = int(gl_GlobalInvocationID.x);
  if (proto_id >= prototype_len) {
    return;
  }

  DrawPrototype proto = prototype_buf[proto_id];
  uint group_id = proto.group_id;
  bool is_inverted = (proto.res_id & 0x80000000u) != 0;
  uint resource_id = (proto.res_id & 0x7FFFFFFFu);

  /* Visibility test result. */
  uint visible_instance_len = 0;
  if (visibility_word_per_draw > 0) {
    uint visibility_word = resource_id * uint(visibility_word_per_draw);
    for (int i = 0; i < visibility_word_per_draw; i++, visibility_word++) {
      /* NOTE: This assumes `proto.instance_len` is 1. */
      /* TODO: Assert. */
      visible_instance_len += uint(bitCount(visibility_buf[visibility_word]));
    }
  }
  else {
    if ((visibility_buf[resource_id / 32u] & (1u << (resource_id % 32u))) != 0) {
      visible_instance_len = proto.instance_len;
    }
  }
  bool is_visible = visible_instance_len > 0;

  /* WORKAROUND (WebGPU/Tint): member-wise copy of the non-atomic fields only —
   * a whole-struct load would also load the atomic counters non-atomically. */
  DrawGroup group;
  group.next = group_buf[group_id].next;
  group.start = group_buf[group_id].start;
  group.len = group_buf[group_id].len;
  group.front_facing_len = group_buf[group_id].front_facing_len;
  group.vertex_len = group_buf[group_id].vertex_len;
  group.vertex_first = group_buf[group_id].vertex_first;
  group.base_index = group_buf[group_id].base_index;

  if (!is_visible) {
    /* Skip the draw but still count towards the completion. */
    if (atomicAddAndGet(group_buf[group_id].total_counter, proto.instance_len) == group.len) {
      write_draw_call(group, group_id);
    }
    return;
  }

  uint back_facing_len = (group.len - group.front_facing_len) * uint(view_len);
  uint dst_index = group.start * uint(view_len);
  if (is_inverted) {
    uint offset = atomicAdd(group_buf[group_id].back_facing_counter, visible_instance_len);
    dst_index += offset;
    if (atomicAddAndGet(group_buf[group_id].total_counter, proto.instance_len) == group.len) {
      write_draw_call(group, group_id);
    }
  }
  else {
    uint offset = atomicAdd(group_buf[group_id].front_facing_counter, visible_instance_len);
    dst_index += back_facing_len + offset;
    if (atomicAddAndGet(group_buf[group_id].total_counter, proto.instance_len) == group.len) {
      write_draw_call(group, group_id);
    }
  }

  /* Fill resource_id buffer for each instance of this draw. */
  if (visibility_word_per_draw > 0) {
    uint visibility_word = resource_id * uint(visibility_word_per_draw);
    for (int i = 0; i < visibility_word_per_draw; i++, visibility_word++) {
      uint word = visibility_buf[visibility_word];
      uint view_index = uint(i) * 32u;
      while (word != 0u) {
        if ((word & 1u) != 0u) {
          if (use_custom_ids) {
            resource_id_buf[dst_index * 2] = view_index | (resource_id << view_shift);
            resource_id_buf[dst_index * 2 + 1] = proto.custom_id;
          }
          else {
            resource_id_buf[dst_index] = view_index | (resource_id << view_shift);
          }
          dst_index++;
        }
        view_index++;
        word >>= 1u;
      }
    }
  }
  else {
    for (uint i = dst_index; i < dst_index + visible_instance_len; i++) {
      if (use_custom_ids) {
        resource_id_buf[i * 2] = resource_id;
        resource_id_buf[i * 2 + 1] = proto.custom_id;
      }
      else {
        resource_id_buf[i] = resource_id;
      }
    }
  }
}
