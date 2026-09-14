/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup GHOST
 */

#include <cstring>

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>

#include "GHOST_EventButton.hh"
#include "GHOST_EventCursor.hh"
#include "GHOST_EventKey.hh"
#include "GHOST_EventWheel.hh"
#include "GHOST_SystemWeb.hh"
#include "GHOST_TimerManager.hh"
#include "GHOST_WindowManager.hh"

#define WEB_CANVAS "#canvas"

/* Cap on how many device pixels we are willing to draw per CSS pixel. The cost
 * is quadratic and a phone can report 3 or 4; 2 is where it stops being worth
 * it. */
#define WEB_MAX_PIXEL_RATIO 2.0

/* Device pixels per CSS pixel for this display. Everything GHOST reports to
 * Blender is in device pixels, so the framebuffer matches the screen instead of
 * being drawn at CSS resolution and stretched by the browser. */
static double web_pixel_ratio()
{
  const double ratio = emscripten_get_device_pixel_ratio();
  if (!(ratio > 0.0)) {
    return 1.0;
  }
  return (ratio > WEB_MAX_PIXEL_RATIO) ? WEB_MAX_PIXEL_RATIO : ratio;
}

/* Read the canvas CSS box and turn it into a device-pixel size, resizing the
 * canvas backing store to match. Returns false when the page has not laid the
 * canvas out yet. */
static bool web_canvas_device_size(uint32_t *r_width, uint32_t *r_height, double *r_ratio)
{
  double css_w = 0.0, css_h = 0.0;
  if (emscripten_get_element_css_size(WEB_CANVAS, &css_w, &css_h) != EMSCRIPTEN_RESULT_SUCCESS ||
      css_w <= 0.0 || css_h <= 0.0)
  {
    return false;
  }
  const double ratio = web_pixel_ratio();
  const uint32_t width = uint32_t(css_w * ratio + 0.5);
  const uint32_t height = uint32_t(css_h * ratio + 0.5);
  /* The backing store is ours to set: the page sizes the element in CSS pixels
   * and must not also own its resolution. */
  emscripten_set_canvas_element_size(WEB_CANVAS, int(width), int(height));
  *r_width = width;
  *r_height = height;
  *r_ratio = ratio;
  return true;
}

/* --- Static trampolines ----------------------------------------------------- */

static bool web_mouse_cb(int type, const EmscriptenMouseEvent *e, void *user)
{
  return static_cast<GHOST_SystemWeb *>(user)->handleMouse(type, e);
}
static bool web_wheel_cb(int /*type*/, const EmscriptenWheelEvent *e, void *user)
{
  return static_cast<GHOST_SystemWeb *>(user)->handleWheel(e);
}
static bool web_key_cb(int type, const EmscriptenKeyboardEvent *e, void *user)
{
  return static_cast<GHOST_SystemWeb *>(user)->handleKey(type, e);
}
static bool web_resize_cb(int /*type*/, const EmscriptenUiEvent * /*e*/, void *user)
{
  return static_cast<GHOST_SystemWeb *>(user)->handleResize();
}

/* --- Key mapping ------------------------------------------------------------ */

static GHOST_TKey web_key_from_code(const char *code)
{
  /* `code` is the PHYSICAL key (KeyboardEvent.code). */
  if (strncmp(code, "Key", 3) == 0 && code[3] >= 'A' && code[3] <= 'Z' && code[4] == '\0') {
    return GHOST_TKey(int(GHOST_kKeyA) + (code[3] - 'A'));
  }
  if (strncmp(code, "Digit", 5) == 0 && code[5] >= '0' && code[5] <= '9' && code[6] == '\0') {
    return GHOST_TKey(int(GHOST_kKey0) + (code[5] - '0'));
  }
  if (strncmp(code, "Numpad", 6) == 0 && code[6] >= '0' && code[6] <= '9' && code[7] == '\0') {
    return GHOST_TKey(int(GHOST_kKeyNumpad0) + (code[6] - '0'));
  }
  if (code[0] == 'F' && code[1] >= '1' && code[1] <= '9') {
    int num = atoi(code + 1);
    if (num >= 1 && num <= 24) {
      return GHOST_TKey(int(GHOST_kKeyF1) + num - 1);
    }
  }
  struct {
    const char *code;
    GHOST_TKey key;
  } table[] = {
      {"Space", GHOST_kKeySpace},
      {"Enter", GHOST_kKeyEnter},
      {"NumpadEnter", GHOST_kKeyNumpadEnter},
      {"Escape", GHOST_kKeyEsc},
      {"Tab", GHOST_kKeyTab},
      {"Backspace", GHOST_kKeyBackSpace},
      {"Delete", GHOST_kKeyDelete},
      {"Insert", GHOST_kKeyInsert},
      {"Home", GHOST_kKeyHome},
      {"End", GHOST_kKeyEnd},
      {"PageUp", GHOST_kKeyUpPage},
      {"PageDown", GHOST_kKeyDownPage},
      {"ArrowLeft", GHOST_kKeyLeftArrow},
      {"ArrowRight", GHOST_kKeyRightArrow},
      {"ArrowUp", GHOST_kKeyUpArrow},
      {"ArrowDown", GHOST_kKeyDownArrow},
      {"ShiftLeft", GHOST_kKeyLeftShift},
      {"ShiftRight", GHOST_kKeyRightShift},
      {"ControlLeft", GHOST_kKeyLeftControl},
      {"ControlRight", GHOST_kKeyRightControl},
      {"AltLeft", GHOST_kKeyLeftAlt},
      {"AltRight", GHOST_kKeyRightAlt},
      {"MetaLeft", GHOST_kKeyLeftOS},
      {"MetaRight", GHOST_kKeyRightOS},
      {"Minus", GHOST_kKeyMinus},
      {"Equal", GHOST_kKeyEqual},
      {"BracketLeft", GHOST_kKeyLeftBracket},
      {"BracketRight", GHOST_kKeyRightBracket},
      {"Backslash", GHOST_kKeyBackslash},
      {"Semicolon", GHOST_kKeySemicolon},
      {"Quote", GHOST_kKeyQuote},
      {"Backquote", GHOST_kKeyAccentGrave},
      {"Comma", GHOST_kKeyComma},
      {"Period", GHOST_kKeyPeriod},
      {"Slash", GHOST_kKeySlash},
      {"CapsLock", GHOST_kKeyCapsLock},
      {"NumpadAdd", GHOST_kKeyNumpadPlus},
      {"NumpadSubtract", GHOST_kKeyNumpadMinus},
      {"NumpadMultiply", GHOST_kKeyNumpadAsterisk},
      {"NumpadDivide", GHOST_kKeyNumpadSlash},
      {"NumpadDecimal", GHOST_kKeyNumpadPeriod},
  };
  for (const auto &entry : table) {
    if (strcmp(code, entry.code) == 0) {
      return entry.key;
    }
  }
  return GHOST_kKeyUnknown;
}

/* --- System ------------------------------------------------------------------ */

GHOST_SystemWeb::GHOST_SystemWeb() : GHOST_System() {}

GHOST_TSuccess GHOST_SystemWeb::init()
{
  GHOST_TSuccess success = GHOST_System::init();
  if (!success) {
    return GHOST_kFailure;
  }

  /* Canvas size: the page owns the CSS box, we own the resolution. */
  uint32_t width = 0, height = 0;
  double ratio = 1.0;
  if (web_canvas_device_size(&width, &height, &ratio)) {
    canvas_w_ = width;
    canvas_h_ = height;
    pixel_ratio_ = ratio;
  }

  /* Input callbacks, delivered on this (the Blender main) thread whenever it
   * yields to its event loop. */
  emscripten_set_mousedown_callback(WEB_CANVAS, this, false, web_mouse_cb);
  emscripten_set_mouseup_callback(WEB_CANVAS, this, false, web_mouse_cb);
  emscripten_set_mousemove_callback(WEB_CANVAS, this, false, web_mouse_cb);
  emscripten_set_wheel_callback(WEB_CANVAS, this, false, web_wheel_cb);
  emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, this, false, web_key_cb);
  emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, this, false, web_key_cb);
  emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, this, false, web_resize_cb);
  return GHOST_kSuccess;
}

uint64_t GHOST_SystemWeb::getMilliSeconds() const
{
  return uint64_t(emscripten_get_now());
}

bool GHOST_SystemWeb::processEvents(bool /*waitForEvent*/)
{
  /* html5 callbacks have already pushed into the event queue (they run when the
   * main-loop iteration yields). Fire timers too. */
  bool has_event = false;
  if (timer_manager_) {
    has_event |= timer_manager_->fireTimers(getMilliSeconds());
  }
  has_event |= (event_manager_->getNumEvents() > 0);
  return has_event;
}

GHOST_IWindow *GHOST_SystemWeb::createWindow(const char *title,
                                             int32_t /*left*/,
                                             int32_t /*top*/,
                                             uint32_t /*width*/,
                                             uint32_t /*height*/,
                                             GHOST_TWindowState state,
                                             GHOST_GPUSettings gpu_settings,
                                             const bool /*exclusive*/,
                                             const bool /*is_dialog*/,
                                             const GHOST_IWindow * /*parent_window*/)
{
  if (window_ != nullptr) {
    /* Single-window platform: additional windows (preferences, etc.) are not
     * supported; return null so WM falls back to opening in the main window. */
    return nullptr;
  }
  const GHOST_ContextParams context_params = GHOST_CONTEXT_PARAMS_FROM_GPU_SETTINGS(gpu_settings);
  window_ = new GHOST_WindowWeb(title, canvas_w_, canvas_h_, state, context_params, pixel_ratio_);
  /* Register with the window manager so validWindow() passes — otherwise every
   * GHOST event is dropped as "invalid window" in wm_window's event handler. */
  window_manager_->addWindow(window_);
  window_manager_->setActiveWindow(window_);
  pushEvent(std::unique_ptr<const GHOST_IEvent>(
      new GHOST_Event(getMilliSeconds(), GHOST_kEventWindowSize, window_)));
  return window_;
}

/* --- Event handlers ----------------------------------------------------------- */

bool GHOST_SystemWeb::handleMouse(int event_type, const EmscriptenMouseEvent *e)
{
  if (window_ == nullptr) {
    return false;
  }
  const uint64_t t = getMilliSeconds();
  /* Browser events are in CSS pixels; the window is in device pixels. */
  cursor_x_ = int32_t(double(e->targetX) * pixel_ratio_);
  cursor_y_ = int32_t(double(e->targetY) * pixel_ratio_);

  modifiers_.set(GHOST_kModifierKeyLeftShift, e->shiftKey);
  modifiers_.set(GHOST_kModifierKeyLeftControl, e->ctrlKey);
  modifiers_.set(GHOST_kModifierKeyLeftAlt, e->altKey);

  switch (event_type) {
    case EMSCRIPTEN_EVENT_MOUSEMOVE:
      pushEvent(std::unique_ptr<const GHOST_IEvent>(new GHOST_EventCursor(
          t, GHOST_kEventCursorMove, window_, cursor_x_, cursor_y_, GHOST_TABLET_DATA_NONE)));
      return true;
    case EMSCRIPTEN_EVENT_MOUSEDOWN:
    case EMSCRIPTEN_EVENT_MOUSEUP: {
      GHOST_TButton button = (e->button == 0) ? GHOST_kButtonMaskLeft :
                             (e->button == 1) ? GHOST_kButtonMaskMiddle :
                                                GHOST_kButtonMaskRight;
      const bool down = (event_type == EMSCRIPTEN_EVENT_MOUSEDOWN);
      buttons_.set(button, down);
      pushEvent(std::unique_ptr<const GHOST_IEvent>(new GHOST_EventButton(t,
                                      down ? GHOST_kEventButtonDown : GHOST_kEventButtonUp,
                                      window_,
                                      button,
                                      GHOST_TABLET_DATA_NONE)));
      return true;
    }
    default:
      return false;
  }
}

bool GHOST_SystemWeb::handleWheel(const EmscriptenWheelEvent *e)
{
  if (window_ == nullptr || e->deltaY == 0.0) {
    return false;
  }
  const int32_t value = (e->deltaY > 0) ? -1 : 1;
  pushEvent(std::unique_ptr<const GHOST_IEvent>(new GHOST_EventWheel(getMilliSeconds(), window_, GHOST_kEventWheelAxisVertical, value)));
  return true;
}

bool GHOST_SystemWeb::handleKey(int event_type, const EmscriptenKeyboardEvent *e)
{
  if (window_ == nullptr) {
    return false;
  }
  const GHOST_TKey key = web_key_from_code(e->code);
  if (key == GHOST_kKeyUnknown) {
    return false;
  }
  const bool down = (event_type == EMSCRIPTEN_EVENT_KEYDOWN);

  switch (key) {
    case GHOST_kKeyLeftShift:
    case GHOST_kKeyRightShift:
      modifiers_.set(GHOST_kModifierKeyLeftShift, down);
      break;
    case GHOST_kKeyLeftControl:
    case GHOST_kKeyRightControl:
      modifiers_.set(GHOST_kModifierKeyLeftControl, down);
      break;
    case GHOST_kKeyLeftAlt:
    case GHOST_kKeyRightAlt:
      modifiers_.set(GHOST_kModifierKeyLeftAlt, down);
      break;
    default:
      break;
  }

  char utf8[6] = {0};
  if (down && e->key[0] != '\0' && e->key[1] == '\0') {
    /* Single-character `key` value = printable glyph. */
    utf8[0] = e->key[0];
  }
  if (getenv("GHOST_LOG_KEYS")) {
    fprintf(stderr,
            "GHOST_KEY %s code='%s' key='%s' ghost=%d utf8='%s'\n",
            down ? "dn" : "up",
            e->code,
            e->key,
            int(key),
            utf8);
    fflush(stderr);
  }
  pushEvent(std::unique_ptr<const GHOST_IEvent>(new GHOST_EventKey(getMilliSeconds(),
                               down ? GHOST_kEventKeyDown : GHOST_kEventKeyUp,
                               window_,
                               key,
                               e->repeat,
                               utf8)));
  /* Let the browser keep F5/F11/devtools shortcuts etc. only when unhandled. */
  return true;
}

bool GHOST_SystemWeb::handleResize()
{
  uint32_t width = 0, height = 0;
  double ratio = 1.0;
  if (!web_canvas_device_size(&width, &height, &ratio)) {
    return false;
  }
  canvas_w_ = width;
  canvas_h_ = height;
  pixel_ratio_ = ratio;
  if (window_) {
    window_->resize(canvas_w_, canvas_h_, pixel_ratio_);
    pushEvent(std::unique_ptr<const GHOST_IEvent>(new GHOST_Event(getMilliSeconds(), GHOST_kEventWindowSize, window_)));
  }
  return true;
}
