"""
Motion-Controlled Subway Surfers Controller
===========================================

Real-time body-tracking controller for *Subway Surfers* using:
  * OpenCV            -> webcam capture + on-screen overlay
  * MediaPipe Pose    -> body landmark detection (nose / wrists)
  * PyAutoGUI         -> keyboard events sent to the game
  * collections.deque -> rolling motion buffer for pause detection

Controls
--------
  * Lean LEFT / RIGHT  -> moves the runner between the 3 lanes.
  * JUMP  (rise fast)  -> 'up' arrow.
  * DUCK  (crouch)     -> 'down' arrow.
  * Stand still / leave frame -> auto-PAUSE ('esc').

Startup uses a two-phase automatic countdown (no manual threshold picking):
  Phase 1 "stand"  (5 s) - "PLEASE TAKE YOUR FAVORITE STAND": learns your
              neutral nose-Y as ``baseline_y`` and derives the up/down/base
              thresholds from your head height (nose<->shoulder gap), so the
              script adapts to your height and camera distance automatically.
  Phase 2 "start"  (5 s) - "START THE GAME": time to focus the game window.
  Phase 3 "playing"       - normal control begins.

Set ``ENABLE_GAME_CONTROL = False`` to run in *simulation* mode:
the video + full detection pipeline run, but no keys are pressed -
detected actions are only printed to the console.  Handy for tuning
thresholds without spamming a real game.

Quit the window with the 'q' key (ESC is reserved as the in-game
pause key, so it is intentionally *not* bound to closing the window).
"""

import sys
import time
import types
from collections import deque
from statistics import median

import cv2
import numpy as np
import mediapipe as mp
import pyautogui


# ============================================================================
# CONFIGURATION  -  tune everything here
# ============================================================================

# --- master switch -----------------------------------------------------------
ENABLE_GAME_CONTROL: bool = True      # True = send real key presses,
                                      # False = simulation (print only)

# --- camera --------------------------------------------------------------
CAMERA_INDEX: int = 0                 # laptop webcam (0 = built-in camera)
CAMERA_BACKEND: int = cv2.CAP_DSHOW   # DirectShow = most stable on Windows
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

# --- calibration -------------------------------------------------------------
# Automatic two-phase countdown calibration (no manual threshold picking):
#   Phase 1 "stand":  5s countdown - "Please take your favorite stand".
#                     Nose-Y samples are collected; at the end baseline_y = median.
#   Phase 2 "start":  5s countdown - "Start the game" (time to focus the game).
#   Phase 3 "playing": normal control begins.
#
# Up/down thresholds are derived automatically from the player's HEAD HEIGHT
# (distance nose -> shoulder midpoint in normalized coords).  This adapts to
# both the player's height and their distance from the camera with zero input.
STAND_COUNTDOWN_SECONDS: float = 5.0    # phase 1: learn the neutral stand pose
START_COUNTDOWN_SECONDS: float = 5.0    # phase 2: "start the game" grace
# thresholds as a fraction of head-height (nose<->shoulder-mid vertical gap):
JUMP_FRACTION_OF_HEAD: float = 0.55     # nose must rise ~55% of head-height to jump
DUCK_FRACTION_OF_HEAD: float = 0.55     # nose must drop  ~55% of head-height to duck
BASE_FRACTION_OF_HEAD: float = 0.28     # neutral band ~28% of head-height around base

# --- Y-axis gestures (normalized 0-1 coordinates) ----------------------------
# These are *defaults*; the automatic countdown calibration (and the interactive
# 'c' wizard) overrides them at runtime to match each user's height & camera
# distance.
BASE_THRESHOLD: float = 0.06          # neutral "base" band around baseline_y
JUMP_THRESHOLD: float = 0.10          # nose must rise past base by this much to jump
DUCK_THRESHOLD: float = 0.10          # nose must drop by this much to roll (down)
Y_KEY_COOLDOWN: float = 0.45          # seconds between jump/down presses

# --- latency tuning (keyboard event shape) -----------------------------------
# pyautogui.press() defaults to an 80ms down+up hold, which is the main cause
# of "the screen reacts instantly but the game lags".  Shorter holds send the
# key faster, and key_down/key_up pairs avoid the fixed press() delay entirely.
KEY_HOLD_SECONDS: float = 0.02        # how long a key is held down (low = snappy)
USE_KEY_DOWN_UP: bool = True          # True = keyDown/keyUp pair (fastest path)

# --- X-axis 3-lane state machine (shoulder-based) -----------------------------
# Lane position is the MIDPOINT of both shoulders, not the nose.  Shoulders
# are wide, stable landmarks that move with your whole body, so lane changes
# are detected the moment your body shifts - no "I moved but I'm still in the
# center split" delay.
#
# The center zone is intentionally narrow (0.40-0.60) so you only need to
# shift part of your body to enter the left/right zone - you don't have to
# move your whole body all the way over.
LEFT_BOUNDARY: float = 0.40         # shoulder-midpoint x < 0.40 -> lane 0 (LEFT)
RIGHT_BOUNDARY: float = 0.60          # 0.40<=x<=0.60               -> lane 1 (CENTER)
                                      # x > 0.60                    -> lane 2 (RIGHT)
LANE_HYSTERESIS: float = 0.025        # small dead-band; flicker rejection (narrow center)
LANE_SMOOTH_ALPHA: float = 1.0        # 1.0 = RAW shoulder-x (instant 2-press transitions)

# --- motion / auto-pause -----------------------------------------------------
MOTION_HISTORY_SIZE: int = 20         # frames of landmark history kept (deque)
MOTION_SMOOTH_ALPHA: float = 0.5      # EMA weight for the per-frame movement level
MOTION_THRESHOLD: float = 0.012       # smoothed movement below this -> "still" (low = active easily)
MIN_VISIBILITY: float = 0.5           # landmark confidence floor
STILL_FRAMES_TO_PAUSE: int = 60       # ~2s at 30fps of genuine stillness before auto-pause
MOVE_FRAMES_TO_RESUME: int = 2        # consecutive moving frames before resume timer starts
RESUME_SECONDS: float = 3.0           # must be moving for this many seconds before esc resume fires
PAUSE_COOLDOWN: float = 0.3           # seconds between pause/resume toggles (low = responsive)
POST_CALIBRATION_GRACE: float = 10.0  # seconds after calibration before auto-pause can fire
GESTURE_FLASH_SECONDS: float = 0.6    # how long the big UP!/DOWN! label stays on

# --- MediaPipe Pose ----------------------------------------------------------
POSE_MIN_DETECTION_CONFIDENCE: float = 0.6
POSE_MIN_TRACKING_CONFIDENCE: float = 0.6
POSE_MODEL_COMPLEXITY: int = 1        # 0=lite, 1=full, 2=heavy

# --- misc --------------------------------------------------------------------
WINDOW_NAME: str = "Subway Surfers - Motion Controller"
QUIT_KEY: str = "q"                   # press 'q' in the video window to exit


# ============================================================================
# Landmark indices we care about (MediaPipe Pose solution)
# ============================================================================
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16


# ============================================================================
# Small drawing / maths helpers
# ============================================================================

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash=10, gap=10):
    """OpenCV has no built-in dashed line, so we draw it as segments.

    This version only needs the *vertical* case (both ends share the same X),
    which is all we use for the lane dividers, but a general Bresenham-style
    fallback is included so the helper stays reusable.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    if x1 == x2:                      # fast path - perfectly vertical line
        y = y1
        step = 1 if y2 >= y1 else -1
        while (y - y2) * step < 0:
            end = y + dash * step
            if (end - y2) * step > 0:
                end = y2
            cv2.line(img, (x1, y), (x2, end), color, thickness, cv2.LINE_AA)
            y = end + gap * step
        return

    # general dashed line (fallback)
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        p_a = (int(x1 + dx * pos), int(y1 + dy * pos))
        p_b = (int(x1 + dx * end), int(y1 + dy * end))
        cv2.line(img, p_a, p_b, color, thickness, cv2.LINE_AA)
        pos = end + gap


def ema(previous: float, current: float, alpha: float) -> float:
    """Exponential moving average used to smooth the raw nose-x signal."""
    return alpha * current + (1.0 - alpha) * previous


def _put_text(img, text, org, scale, color, thickness=2, bg=(0, 0, 0)):
    """Text with a subtle dark shadow so it stays readable on any feed."""
    x, y = org
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, bg, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ============================================================================
# Controller
# ============================================================================

class MotionController:
    """Holds every piece of mutable state for the control loop.

    Keeping state inside a class (instead of globals) makes the logic easy to
    unit-test and keeps the per-frame function signatures tiny.
    """

    def __init__(self):
        # calibration ----------------------------------------------------
        # Automatic countdown phases: "stand" -> "start" -> "playing"
        self.phase: str = "stand"             # current calibration/control phase
        self._phase_start: float = time.perf_counter()
        self._stand_samples: list[float] = [] # nose-Y samples during stand phase
        self._head_height: float | None = None  # nose<->shoulder-mid gap (for auto thresholds)
        self.baseline_y: float | None = None
        self.calibrated: bool = False
        self._calibrated_at: float = 0.0    # timestamp when calibration finished

        # lane state machine ---------------------------------------------
        self.current_lane: int = 1          # start in the CENTER lane
        self.smooth_x: float | None = None  # EMA of shoulder-midpoint x
        self.shoulder_mid_x: float | None = None  # latest shoulder midpoint (for HUD)

        # gesture cooldowns ----------------------------------------------
        self._last_jump: float = 0.0
        self._last_duck: float = 0.0
        self._vertical_ready: bool = True  # must return to baseline between gestures

        # motion / pause ---------------------------------------------------
        self.motion_history: deque = deque(maxlen=MOTION_HISTORY_SIZE)
        self._motion_level: float | None = None  # EMA-smoothed movement level
        self._last_left_y: float = 0.5    # persisted so occluded wrists don't spike
        self._last_right_y: float = 0.5
        self._still_streak: int = 0
        self._move_streak: int = 0
        self._resume_move_start: float = 0.0   # timestamp when sustained resume motion began
        self.paused: bool = False
        self._last_pause_toggle: float = 0.0

        # bookkeeping / debug ----------------------------------------------
        self.last_action: str = "CALIBRATING"
        self.frame_times: deque = deque(maxlen=30)   # for FPS
        self.nose_x: float | None = None  # latest nose position (for the HUD)
        self.nose_y: float | None = None
        self.gesture_flash: tuple | None = None      # ("JUMP"/"DOWN", timestamp)

    # ------------------------------------------------------------------ #
    # 1. Calibration  (automatic two-phase countdown)
    # ------------------------------------------------------------------ #
    def _head_height_from(self, nose, landmarks) -> float:
        """Estimate head height = vertical gap nose <-> shoulder midpoint.

        In normalized image coords y grows downward, so the nose sits *above*
        the shoulders (smaller y).  head_height = shoulder_mid_y - nose.y.
        Falls back to a small constant if shoulders aren't visible.
        """
        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]
        mids = []
        if ls.visibility >= MIN_VISIBILITY:
            mids.append(ls.y)
        if rs.visibility >= MIN_VISIBILITY:
            mids.append(rs.y)
        if mids:
            shoulder_mid_y = sum(mids) / len(mids)
            return max(shoulder_mid_y - nose.y, 0.05)   # clamp to a sane min
        return 0.12   # fallback head-height when shoulders occluded

    def _apply_head_height_thresholds(self, head_height: float) -> None:
        """Derive BASE/JUMP/DUCK thresholds from the player's head height.

        This auto-adapts to height and camera distance: a tall person close to
        the camera has a large head-height in normalized coords, so the
        thresholds scale up; a short / far person scales down.
        """
        global BASE_THRESHOLD, JUMP_THRESHOLD, DUCK_THRESHOLD
        BASE_THRESHOLD = head_height * BASE_FRACTION_OF_HEAD
        JUMP_THRESHOLD = head_height * JUMP_FRACTION_OF_HEAD
        DUCK_THRESHOLD = head_height * DUCK_FRACTION_OF_HEAD
        print(f"[CALIBRATION] head_height={head_height:.3f} -> "
              f"BASE={BASE_THRESHOLD:.3f} JUMP={JUMP_THRESHOLD:.3f} DUCK={DUCK_THRESHOLD:.3f}")

    def run_stand_phase(self, nose, landmarks) -> str:
        """Phase 1: 5s countdown collecting the neutral stand pose.

        Returns the HUD action string for this frame.
        """
        self._stand_samples.append(nose.y)
        # keep a running head-height estimate (use the latest visible sample)
        self._head_height = self._head_height_from(nose, landmarks)
        elapsed = time.perf_counter() - self._phase_start
        if elapsed >= STAND_COUNTDOWN_SECONDS and self._stand_samples:
            self.baseline_y = median(self._stand_samples)
            self.calibrated = True
            self._calibrated_at = time.perf_counter()
            # derive up/down thresholds from head height
            if self._head_height is not None:
                self._apply_head_height_thresholds(self._head_height)
            # advance to the "start" countdown phase
            self.phase = "start"
            self._phase_start = time.perf_counter()
            self._stand_samples.clear()
            self._vertical_ready = True
            print(f"[CALIBRATION] stand complete -> baseline_y={self.baseline_y:.3f}")
        return "CALIBRATING"

    def run_start_phase(self) -> str:
        """Phase 2: 5s 'Start the game' countdown before control begins."""
        elapsed = time.perf_counter() - self._phase_start
        if elapsed >= START_COUNTDOWN_SECONDS:
            self.phase = "playing"
            self._calibrated_at = time.perf_counter()   # reset grace period
            self._still_streak = 0
            self.last_action = "RUNNING"
            print("[CALIBRATION] start countdown complete -> game control active")
        return "GET READY"

    @property
    def stand_countdown(self) -> float:
        """Seconds remaining in the stand (phase 1) countdown."""
        if self.phase != "stand":
            return 0.0
        return max(STAND_COUNTDOWN_SECONDS - (time.perf_counter() - self._phase_start), 0.0)

    @property
    def start_countdown(self) -> float:
        """Seconds remaining in the start (phase 2) countdown."""
        if self.phase != "start":
            return 0.0
        return max(START_COUNTDOWN_SECONDS - (time.perf_counter() - self._phase_start), 0.0)

    @property
    def calibration_progress(self) -> float:
        """0.0 -> 1.0 progress of the *current* countdown phase (for the HUD bar)."""
        if self.phase == "stand":
            return min((time.perf_counter() - self._phase_start) / STAND_COUNTDOWN_SECONDS, 1.0)
        if self.phase == "start":
            return min((time.perf_counter() - self._phase_start) / START_COUNTDOWN_SECONDS, 1.0)
        return 1.0

    # ------------------------------------------------------------------ #
    # 2. X-axis - 3 lane state machine
    # ------------------------------------------------------------------ #
    @staticmethod
    def classify_lane(x_norm: float) -> int:
        """Map a normalized x in [0,1] to lane 0 (LEFT), 1 (CENTER), 2 (RIGHT)."""
        if x_norm < LEFT_BOUNDARY:
            return 0
        if x_norm > RIGHT_BOUNDARY:
            return 2
        return 1

    def _resolve_lane(self, raw_x: float) -> int:
        """Smooth nose-x and apply hysteresis.  Does not send keys."""
        self.smooth_x = raw_x if self.smooth_x is None else ema(
            self.smooth_x, raw_x, LANE_SMOOTH_ALPHA)

        target = self.classify_lane(self.smooth_x)

        # hysteresis - only accept a lane change when we are clearly past a
        # boundary, otherwise keep the current lane.
        if target != self.current_lane:
            if self.current_lane == 0 and not (self.smooth_x > LEFT_BOUNDARY + LANE_HYSTERESIS):
                target = self.current_lane
            elif self.current_lane == 2 and not (self.smooth_x < RIGHT_BOUNDARY - LANE_HYSTERESIS):
                target = self.current_lane
            elif self.current_lane == 1:
                if target == 0 and not (self.smooth_x < LEFT_BOUNDARY - LANE_HYSTERESIS):
                    target = self.current_lane
                elif target == 2 and not (self.smooth_x > RIGHT_BOUNDARY + LANE_HYSTERESIS):
                    target = self.current_lane
        return target

    def apply_lane_change(self, raw_x: float) -> None:
        """Smooth the signal, work out the target lane and press arrows."""
        target = self._resolve_lane(raw_x)
        lane_diff = target - self.current_lane
        if lane_diff > 0:
            for _ in range(lane_diff):
                self._press("right")
        elif lane_diff < 0:
            for _ in range(abs(lane_diff)):
                self._press("left")
        self.current_lane = target

    def sync_lane_without_keys(self, raw_x: float) -> None:
        """Track the body lane without sending catch-up arrow presses."""
        self.current_lane = self._resolve_lane(raw_x)

    # ------------------------------------------------------------------ #
    # 3. Y-axis - jump / duck
    # ------------------------------------------------------------------ #
    def classify_vertical(self, nose_y: float) -> str | None:
        """Return 'JUMP' / 'DOWN' / None relative to the calibrated baseline."""
        if not self.calibrated or self.baseline_y is None:
            return None

        now = time.perf_counter()
        delta_up = self.baseline_y - nose_y      # > 0  -> moved UP   (jump)
        delta_down = nose_y - self.baseline_y    # > 0  -> moved DOWN (roll)

        # A held jump/down pose must return inside the BASE band before it can
        # re-fire.  Standing normally stays inside BASE_THRESHOLD, so it never
        # fires a false JUMP.
        if not self._vertical_ready:
            if abs(nose_y - self.baseline_y) < BASE_THRESHOLD:
                self._vertical_ready = True
            else:
                return None

        if delta_up > JUMP_THRESHOLD and now - self._last_jump > Y_KEY_COOLDOWN:
            self._press("up")
            self._last_jump = now
            self._vertical_ready = False
            return "JUMP"
        if delta_down > DUCK_THRESHOLD and now - self._last_duck > Y_KEY_COOLDOWN:
            self._press("down")
            self._last_duck = now
            self._vertical_ready = False
            return "DOWN"
        return None

    # ------------------------------------------------------------------ #
    # 4. motion / auto-pause
    # ------------------------------------------------------------------ #
    def record_motion(self, nose_x: float, nose_y: float,
                      left_y: float | None, right_y: float | None) -> None:
        """Append one motion sample to the rolling deque.

        Occluded wrists (``None``) reuse their last visible value instead of
        jumping to 0.0, so a briefly-lost hand no longer spikes the score.
        """
        if left_y is None:
            left_y = self._last_left_y
        else:
            self._last_left_y = left_y
        if right_y is None:
            right_y = self._last_right_y
        else:
            self._last_right_y = right_y

        # per-frame movement (delta) vs the previous sample, EMA-smoothed so a
        # single noisy frame neither starts nor stops the "moving" state.
        if self.motion_history:
            px, py, pl, pr = self.motion_history[-1]
            move = abs(nose_x - px) + abs(nose_y - py) + abs(left_y - pl) + abs(right_y - pr)
        else:
            move = 0.0
        self._motion_level = move if self._motion_level is None else ema(
            self._motion_level, move, MOTION_SMOOTH_ALPHA)

        self.motion_history.append((nose_x, nose_y, left_y, right_y))

    def record_no_detection(self) -> None:
        """No landmarks this frame -> repeat the last sample (zero movement)."""
        if self.motion_history:
            self.motion_history.append(self.motion_history[-1])
        else:
            self.motion_history.append((0.5, 0.5, 0.5, 0.5))

    def motion_score(self) -> float:
        """Smoothed movement level (0 = perfectly still, larger = more active).

        A *delta* (velocity) metric over nose-x, nose-y and both wrist-y.  It
        counts lane-leaning (nose-x) and jumping (nose-y) as motion, unlike a
        wrist-y-only variance, which is what made "running" unreliable before.
        """
        return float("inf") if self._motion_level is None else self._motion_level

    def is_moving(self) -> bool:
        """True when the motion score is at/above MOTION_THRESHOLD."""
        return self.motion_score() >= MOTION_THRESHOLD

    def update_pause(self, visible: bool) -> bool:
        """Toggle the in-game pause with hysteresis to stop flapping.

        * Pause  -> player leaves the frame, OR stays still for
          ``STILL_FRAMES_TO_PAUSE`` consecutive frames.  Presses ``esc`` once.
        * Resume -> player is visible AND shows a clear burst of movement.
          Presses ``esc`` once to unpause.  Resume is deliberately sensitive
          (a single strong-motion frame) so you can jump straight back in.
        ``esc`` toggles Subway Surfers' pause menu, so we only press on a state
        *transition* (and rate-limit it), never every frame.
        Returns True when a transition happened this frame.
        """
        if not self.calibrated:
            return False

        now = time.perf_counter()
        score = self.motion_score()
        moving = self.is_moving()

        if visible and moving:
            self._move_streak += 1
            self._still_streak = 0
        else:
            self._still_streak += 1
            self._move_streak = 0

        if self.paused:
            # Resume only after the player has been visibly moving for at least
            # RESUME_SECONDS of sustained motion.  This avoids accidental
            # resumes from a single flinch and confirms you really want to
            # continue playing.
            if visible and moving and self._move_streak >= MOVE_FRAMES_TO_RESUME:
                if self._resume_move_start == 0.0:
                    self._resume_move_start = now          # start the timer
                sustained = (now - self._resume_move_start) >= RESUME_SECONDS
                should_pause = not sustained
            else:
                # stopped moving -> reset the resume timer
                self._resume_move_start = 0.0
                should_pause = True
        else:
            # grace period right after calibration: don't auto-pause while the
            # player is getting into position (they were still during calibration)
            in_grace = (now - self._calibrated_at) < POST_CALIBRATION_GRACE
            should_pause = (not in_grace) and (
                (not visible) or (self._still_streak >= STILL_FRAMES_TO_PAUSE))

        if should_pause != self.paused and now - self._last_pause_toggle > PAUSE_COOLDOWN:
            self._press("esc")
            self.paused = should_pause
            self._last_pause_toggle = now
            self._resume_move_start = 0.0                  # clear after toggling
            print(f"[PAUSE] {'paused' if should_pause else 'resumed'} "
                  f"(score={score:.3f} still={self._still_streak} move={self._move_streak})")
            return True
        return False

    # ------------------------------------------------------------------ #
    # 5. input / output plumbing
    # ------------------------------------------------------------------ #
    def _press(self, key: str) -> None:
        """Send a key press, or just log it when simulation mode is on.

        ``pyautogui.press(key, _pause=False)`` is the fastest path on Windows:
        it calls ``keyDown`` + ``keyUp`` via SendInput with **no** internal
        sleep, so it never stalls the main loop.  (Avoid ``press()`` without
        ``_pause=False`` -- it sleeps ~80 ms -- and avoid an explicit
        ``time.sleep`` between keyDown/keyUp, which blocks pose processing.)
        """
        if ENABLE_GAME_CONTROL:
            if USE_KEY_DOWN_UP:
                pyautogui.keyDown(key)
                pyautogui.keyUp(key)           # no sleep -> no loop stall
            else:
                pyautogui.press(key, _pause=False)
        print(f"[{'GAME' if ENABLE_GAME_CONTROL else 'SIM'}] press -> {key}")

    # ------------------------------------------------------------------ #
    # 6. per-frame update
    # ------------------------------------------------------------------ #
    def update(self, landmarks, frame_w: int, frame_h: int) -> str:
        """Run one frame of control logic.  Returns the HUD action string."""
        now = time.perf_counter()
        self.frame_times.append(now)

        nose = landmarks[NOSE]
        visible = nose.visibility >= MIN_VISIBILITY

        # record motion (occluded wrists -> None -> reuse last visible value).
        # A low-confidence nose is treated as no detection so jitter does not
        # inflate the motion score.
        if visible:
            lw = landmarks[LEFT_WRIST]
            rw = landmarks[RIGHT_WRIST]
            left_y = lw.y if lw.visibility >= MIN_VISIBILITY else None
            right_y = rw.y if rw.visibility >= MIN_VISIBILITY else None
            self.record_motion(nose.x, nose.y, left_y, right_y)
            self.nose_x, self.nose_y = nose.x, nose.y

            # lane position = midpoint of both shoulders (falls back to nose-x
            # if shoulders aren't visible).  Shoulders move with the whole body
            # so the lane changes the instant you shift, not when your nose
            # finally crosses a line.
            ls = landmarks[LEFT_SHOULDER]
            rs = landmarks[RIGHT_SHOULDER]
            if ls.visibility >= MIN_VISIBILITY and rs.visibility >= MIN_VISIBILITY:
                self.shoulder_mid_x = (ls.x + rs.x) / 2.0
            elif ls.visibility >= MIN_VISIBILITY:
                self.shoulder_mid_x = ls.x
            elif rs.visibility >= MIN_VISIBILITY:
                self.shoulder_mid_x = rs.x
            else:
                self.shoulder_mid_x = nose.x   # last resort fallback
        else:
            self.record_no_detection()

        # -- calibration phases (automatic countdown) ---------------------
        if self.phase == "stand":
            if visible:
                self.last_action = self.run_stand_phase(nose, landmarks)
            else:
                self.last_action = "CALIBRATING"
            return self.last_action

        if self.phase == "start":
            # "Start the game" countdown: no control actions, just wait.
            self.last_action = self.run_start_phase()
            return self.last_action

        # -- visibility / motion gate ------------------------------------
        if not visible:
            self.update_pause(visible=False)
            self.last_action = "PAUSED"
            return self.last_action

        self.update_pause(visible=True)
        if self.paused:
            # keep lane state aligned with the body without catch-up key spam
            self.sync_lane_without_keys(self.shoulder_mid_x if self.shoulder_mid_x is not None else nose.x)
            self.last_action = "PAUSED"
            return self.last_action

        # -- gestures ------------------------------------------------------
        vertical = self.classify_vertical(nose.y)
        if vertical:
            self.last_action = vertical
            self.gesture_flash = (vertical, now)

        # lane changes use the shoulder midpoint (body position), not the nose
        lane_x = self.shoulder_mid_x if self.shoulder_mid_x is not None else nose.x
        self.apply_lane_change(lane_x)

        if not vertical:
            # distinguish actively-moving play from just standing in view
            self.last_action = "RUNNING" if self.is_moving() else "IDLE"
        return self.last_action

    # ------------------------------------------------------------------ #
    @property
    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / span if span > 0 else 0.0

    @property
    def active_gesture_flash(self) -> str | None:
        """The gesture label to flash ("JUMP"/"DOWN") while it is still fresh."""
        if self.gesture_flash and \
                time.perf_counter() - self.gesture_flash[1] < GESTURE_FLASH_SECONDS:
            return self.gesture_flash[0]
        return None


# ============================================================================
# HUD / overlay
# ============================================================================

LANE_NAMES = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
LANE_COLORS = {0: (255, 200, 0), 1: (0, 255, 0), 2: (0, 165, 255)}
ACTION_COLORS = {
    "JUMP": (0, 255, 255),
    "DOWN": (255, 0, 255),
    "RUNNING": (0, 255, 0),
    "IDLE": (200, 200, 200),
    "PAUSED": (0, 0, 255),
    "CALIBRATING": (200, 200, 200),
}


def draw_zones(frame, w, h):
    """Two dashed vertical dividers + zone captions."""
    for frac in (LEFT_BOUNDARY, RIGHT_BOUNDARY):
        x = int(frac * w)
        draw_dashed_line(frame, (x, 0), (x, h), (255, 255, 255), 2)

    captions = [("LEFT", 0.5 * LEFT_BOUNDARY),
                ("CENTER", 0.5 * (LEFT_BOUNDARY + RIGHT_BOUNDARY)),
                ("RIGHT", 0.5 * (RIGHT_BOUNDARY + 1.0))]
    for name, frac in captions:
        cx = int(frac * w)
        _put_text(frame, name, (cx - 40, 30), 0.7, (255, 255, 255), 2)


def draw_gesture_gauge(frame, ctrl: MotionController, w, h):
    """Vertical gauge (simulation mode) visualising the up/down thresholds.

    Shows the baseline, the JUMP (UP) line above it and the DUCK (DOWN) line
    below it, plus a live marker for the current nose height - so you can see
    exactly when an UP / DOWN gesture will fire.  Screen y grows downward, so
    the UP line sits *above* the baseline on screen.
    """
    if not ctrl.calibrated or ctrl.baseline_y is None or ctrl.nose_y is None:
        return

    gx = w - 60                       # gauge x position (right side)
    top, bottom = 90, h - 170         # gauge pixel range

    def to_pix(y_norm: float) -> int:
        return int(top + (bottom - top) * y_norm)

    base_pix = to_pix(ctrl.baseline_y)
    jump_pix = to_pix(ctrl.baseline_y - JUMP_THRESHOLD)   # above baseline
    down_pix = to_pix(ctrl.baseline_y + DUCK_THRESHOLD)   # below baseline
    nose_pix = to_pix(ctrl.nose_y)
    base_hi_pix = to_pix(ctrl.baseline_y - BASE_THRESHOLD)
    base_lo_pix = to_pix(ctrl.baseline_y + BASE_THRESHOLD)

    # gauge track
    cv2.line(frame, (gx, top), (gx, bottom), (180, 180, 180), 2, cv2.LINE_AA)

    # the UP / DOWN notes are RED while idle and GREEN while the gesture is
    # held, then back to RED once released.
    up_active = (ctrl.baseline_y - ctrl.nose_y) > JUMP_THRESHOLD
    down_active = (ctrl.nose_y - ctrl.baseline_y) > DUCK_THRESHOLD
    up_col = (0, 255, 0) if up_active else (0, 0, 255)
    down_col = (0, 255, 0) if down_active else (0, 0, 255)

    # base (neutral) band = BASE_THRESHOLD above/below the baseline
    cv2.line(frame, (gx - 14, base_hi_pix), (gx + 14, base_hi_pix), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(frame, (gx - 14, base_lo_pix), (gx + 14, base_lo_pix), (120, 120, 120), 1, cv2.LINE_AA)

    # threshold + baseline lines with labels
    cv2.line(frame, (gx - 18, jump_pix), (gx + 18, jump_pix), up_col, 2, cv2.LINE_AA)
    _put_text(frame, "UP", (gx + 24, jump_pix + 6), 0.6, up_col, 2)
    cv2.line(frame, (gx - 18, base_pix), (gx + 18, base_pix), (255, 255, 255), 2, cv2.LINE_AA)
    _put_text(frame, "BASE", (gx + 24, base_pix + 6), 0.5, (255, 255, 255), 1)
    cv2.line(frame, (gx - 18, down_pix), (gx + 18, down_pix), down_col, 2, cv2.LINE_AA)
    _put_text(frame, "DOWN", (gx + 24, down_pix + 6), 0.6, down_col, 2)

    # live nose marker - green while a gesture is held, red while idle
    col = (0, 255, 0) if (up_active or down_active) else (0, 0, 255)
    cv2.circle(frame, (gx, nose_pix), 9, col, -1, cv2.LINE_AA)


def draw_sim_debug(frame, ctrl: MotionController, w, h):
    """Numeric debug readout (simulation only) to help tune the thresholds."""
    du = (ctrl.baseline_y - ctrl.nose_y) if (ctrl.baseline_y is not None and ctrl.nose_y is not None) else 0.0
    dd = -du
    ms = ctrl.motion_score()
    ms_str = "  inf" if ms == float("inf") else f"{ms:.3f}"
    moving = ctrl.is_moving()
    lines = [
        f"dUP  {du:+.3f} / {JUMP_THRESHOLD:.3f}",
        f"dDN  {dd:+.3f} / {DUCK_THRESHOLD:.3f}",
        f"motion {ms_str} / {MOTION_THRESHOLD:.3f} {'MOVE' if moving else 'STILL'}",
        f"streaks still={ctrl._still_streak} move={ctrl._move_streak}",
    ]
    x0, y0 = 10, 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 24), (x0 + 300, y0 + len(lines) * 24), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, ln in enumerate(lines):
        _put_text(frame, ln, (x0, y0 + i * 24), 0.55, (0, 255, 255), 1)


def draw_hud(frame, ctrl: MotionController, w, h):
    """Status panel: calibration, lane, action, FPS, pause, mode."""
    # translucent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - 175), (470, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = h - 145

    # calibration status + progress bar (phase-aware)
    if ctrl.phase == "stand":
        _put_text(frame, "Calibrating...", (20, y), 0.55, (0, 200, 255))
        bar_x0, bar_x1 = 150, 320
        bar_w = int((bar_x1 - bar_x0) * ctrl.calibration_progress)
        cv2.rectangle(frame, (bar_x0, y - 18), (bar_x0 + bar_w, y - 4), (0, 200, 255), -1)
        cv2.rectangle(frame, (bar_x0, y - 18), (bar_x1, y - 4), (255, 255, 255), 1)
    elif ctrl.phase == "start":
        _put_text(frame, "GET READY", (20, y), 0.6, (0, 200, 255))
        bar_x0, bar_x1 = 150, 320
        bar_w = int((bar_x1 - bar_x0) * ctrl.calibration_progress)
        cv2.rectangle(frame, (bar_x0, y - 18), (bar_x0 + bar_w, y - 4), (0, 200, 255), -1)
        cv2.rectangle(frame, (bar_x0, y - 18), (bar_x1, y - 4), (255, 255, 255), 1)
    else:
        _put_text(frame, "CALIBRATED", (20, y), 0.6, (0, 255, 0))

    # active lane
    lane = ctrl.current_lane
    _put_text(frame, f"LANE: {LANE_NAMES[lane]}",
              (20, y + 30), 0.7, LANE_COLORS[lane])

    # last action
    action = ctrl.last_action
    _put_text(frame, f"ACTION: {action}",
              (20, y + 60), 0.7, ACTION_COLORS.get(action, (255, 255, 255)))

    # FPS
    _put_text(frame, f"FPS: {ctrl.fps:4.1f}", (20, y + 90), 0.6, (0, 255, 0))

    # hotkey hints
    _put_text(frame, f"'{QUIT_KEY}'=quit",
              (20, y + 118), 0.45, (180, 180, 180), 1)

    # paused / mode badges (top-right)
    if ctrl.paused:
        _put_text(frame, "PAUSED", (w - 160, 40), 0.9, (0, 0, 255), 3)
    if not ENABLE_GAME_CONTROL:
        _put_text(frame, "SIMULATION", (w - 220, h - 20), 0.7, (0, 200, 255), 2)

    # highlight the active zone with a soft fill
    zone_edges = [0, LEFT_BOUNDARY, RIGHT_BOUNDARY, 1.0]
    x0 = int(zone_edges[lane] * w)
    x1 = int(zone_edges[lane + 1] * w)
    hl = frame.copy()
    cv2.rectangle(hl, (x0, 0), (x1, h), LANE_COLORS[lane], -1)
    cv2.addWeighted(hl, 0.08, frame, 0.92, 0, frame)

    # big flashing UP!/DOWN! label (both modes) so you can see a gesture fire
    flash = ctrl.active_gesture_flash
    if flash:
        col = ACTION_COLORS.get(flash, (255, 255, 255))
        _put_text(frame, f"{flash}!", (w // 2 - 110, h // 2), 1.6, col, 4)

    # simulation-only debugging aids (hidden during real game control)
    if not ENABLE_GAME_CONTROL:
        draw_gesture_gauge(frame, ctrl, w, h)
        draw_sim_debug(frame, ctrl, w, h)


def draw_countdown_banner(frame, ctrl: MotionController, w, h):
    """Big centered countdown + instruction shown during the two startup phases."""
    if ctrl.phase == "stand":
        msg = "PLEASE TAKE YOUR FAVORITE STAND"
        remain = ctrl.stand_countdown
        col = (0, 200, 255)
    elif ctrl.phase == "start":
        msg = "START THE GAME"
        remain = ctrl.start_countdown
        col = (0, 255, 180)
    else:
        return

    # center the message text horizontally
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    _put_text(frame, msg, (w // 2 - tw // 2, h // 2 - th // 2), 0.9, col, 2)

    # center the countdown number horizontally
    num = f"{int(remain) + 1}"
    (nw, nh), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
    _put_text(frame, num, (w // 2 - nw // 2, h // 2 + nh + 20), 1.8, (255, 255, 255), 4)


# ============================================================================
# Video source
# ============================================================================

def open_capture() -> cv2.VideoCapture:
    """Open the laptop webcam via DirectShow.

    Exits with a helpful message if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            print(f"[INFO] Opened camera index={CAMERA_INDEX} backend={CAMERA_BACKEND}")
            return cap
    cap.release()

    sys.exit(
        "[ERROR] Could not open the camera.\n"
        "        Troubleshooting:\n"
        "          * Close apps that may be using the webcam (browser tabs,\n"
        "            Teams/Zoom, the Windows Camera app, OBS, ...).\n"
        "          * Windows Settings > Privacy > Camera: allow desktop apps.\n"
        "          * Make sure the laptop camera is not physically disabled."
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    cap = open_capture()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    ctrl = MotionController()
    mp_pose = mp.solutions.pose

    mode = "GAME CONTROL" if ENABLE_GAME_CONTROL else "SIMULATION"
    print(f"[START] Motion controller running in {mode} mode. "
          f"Press '{QUIT_KEY}' in the video window to quit.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=POSE_MODEL_COMPLEXITY,
        enable_segmentation=False,
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # mirror the image so moving to *your* right moves right on screen
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False            # small perf win
            results = pose.process(rgb)

            if results.pose_landmarks:
                action = ctrl.update(results.pose_landmarks.landmark, w, h)
            else:
                # nobody in frame -> treat as "left the frame" => pause
                # (only during active play so we do not send esc during countdowns)
                ctrl.record_no_detection()
                if ctrl.phase == "playing":
                    ctrl.update_pause(visible=False)
                    action = "PAUSED"
                elif ctrl.phase == "start":
                    action = "GET READY"
                else:
                    action = "CALIBRATING"
                ctrl.last_action = action
                ctrl.frame_times.append(time.perf_counter())

            # ---- overlay ----
            draw_zones(frame, w, h)
            draw_hud(frame, ctrl, w, h)
            draw_countdown_banner(frame, ctrl, w, h)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(QUIT_KEY):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[STOP] Controller shut down.")


if __name__ == "__main__":
    main()

