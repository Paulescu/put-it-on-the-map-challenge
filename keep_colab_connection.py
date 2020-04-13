from pyautogui import press, typewrite, hotkey
import time
from random import shuffle

array = ["left", "right", "up", "down"]

while True:
    shuffle(array)
    time.sleep(10)
    press('left')
    press('right')
    press('up')
    press('down')