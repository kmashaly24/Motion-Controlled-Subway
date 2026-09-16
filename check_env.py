"""Environment sanity check for the controller's dependencies."""
import numpy
print("numpy", numpy.__version__)

import cv2
print("cv2", cv2.__version__)

import pyautogui
print("pyautogui", pyautogui.__version__)

# Stub optional mediapipe.tasks (avoids pulling broken/unused TensorFlow).
import sys, types
for _n in ("mediapipe.tasks", "mediapipe.tasks.python"):
    _s = types.ModuleType(_n); _s.__path__ = []; sys.modules[_n] = _s

import mediapipe as mp
print("mediapipe", mp.__version__)
print("has mp.solutions.pose:", hasattr(mp.solutions, "pose"))

# Build the actual Pose pipeline (validates C++ bindings + bundled model).
pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
pose.close()
print("POSE_CONSTRUCT_OK")

print("ALL_DEPS_OK")
