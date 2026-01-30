#!/usr/bin/env python3
import subprocess
import time
from pynput import keyboard

# =====================
# CONFIG
# =====================
HOVER_SPEED = 650
STEP = 20          # tuş başına hız değişimi
MAX_SPEED = 750
MIN_SPEED = 580
DT = 0.05          # 20 Hz

current_speed = HOVER_SPEED

# =====================
# KEYBOARD
# =====================
def on_press(key):
    global current_speed
    if key == keyboard.Key.up:
        current_speed += STEP
    elif key == keyboard.Key.down:
        current_speed -= STEP

def on_release(key):
    global current_speed
    if key in (keyboard.Key.up, keyboard.Key.down):
        current_speed = HOVER_SPEED  # TUŞ YOKSA DURSUN

keyboard.Listener(on_press=on_press, on_release=on_release).start()

# =====================
# ACTUATOR
# =====================
def send(speed):
    speed = int(max(MIN_SPEED, min(speed, MAX_SPEED)))
    cmd = f"velocity:[{speed},{speed},{speed},{speed}]"
    subprocess.Popen([
        "gz", "topic",
        "-t", "/quadcopter/command/motor_speed",
        "--msgtype", "gz.msgs.Actuators",
        "-p", cmd
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# =====================
# MAIN LOOP
# =====================
print("[INFO] SIMPLE MANUAL THRUST CONTROL STARTED")

while True:
    send(current_speed)
    print(f"Motor speed: {current_speed}", end="\r")
    time.sleep(DT)
