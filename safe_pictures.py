import mouse
import keyboard as kb
import mss
import mss.tools
import cv2
import numpy as np
from time import sleep
import pyautogui
#import os.path as osPath
from os import path as osPath
import os

targetNumberLoc = { # target number location for 1920x1080 fullscreened game
        "top": 166,
        "left": 1020,
        "width": 33,
        "height": 18
        }

safeCodeLoc = { # roughly the current safe code number location for 1920x1080 fullscreened game
    "top": 398,
    "left": 944,
    "width": 35,
    "height": 25
    }

counter = 1

while True:
    ss = mss.mss()

    tarNumC = np.array(ss.grab(targetNumberLoc))
    safeNumC = np.array(ss.grab(safeCodeLoc))

    tarNumGS = cv2.cvtColor(tarNumC, cv2.COLOR_BGR2GRAY)
    safeNumGS = cv2.cvtColor(safeNumC, cv2.COLOR_BGR2GRAY)

    input("enter")
    fntn = "target_number_" + str(counter) + ".png"
    tnsn = "safe_number_" + str(counter) + ".png"
    cv2.imwrite(fntn, tarNumGS)
    #cv2.imwrite(tnsn, safeNumGS)
    counter = counter + 1