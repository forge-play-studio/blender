# Make an emscripten tool's .js node-executable (prepend shebang + chmod +x).
file(READ "${TOOL_JS}" _content)
string(SUBSTRING "${_content}" 0 2 _head)
if(NOT _head STREQUAL "#!")
  file(WRITE "${TOOL_JS}" "#!/usr/bin/env node\n${_content}")
endif()
file(CHMOD "${TOOL_JS}" PERMISSIONS
  OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
