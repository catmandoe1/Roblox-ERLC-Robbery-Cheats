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

pyautogui.FAILSAFE = False # stops it from crashing when in game



def clickLeft():
    mouse.click("left")

def mouseHold():
    mouse.press(button="left")

def mouseRelease():
    mouse.release(button="left")

def clickRight():
    mouse.click("right")

def prepareMouse():
    #pyautogui.moveTo(670, 315, 0)
    moveMouse(670, 315)

def moveMouse(x, y):
    pyautogui.moveTo(x, y, 0)

def buy():
    moveMouse(1454, 962)
    clickLeft()

def buyTool(x, y):
    moveMouse(x, y)
    sleep(0.5)
    mouseHold()
    sleep(0.25)
    mouseRelease()
    sleep(0.25)
    buy()

def clear():
    os.system('CLS')

def get_hog() : 
    winSize = (25,25)
    blockSize = (9,9)
    blockStride = (4,4)
    cellSize = (9,9)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    #affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def GetGammaTable():
    gammalookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        gammalookUpTable[0,i] = np.clip(pow(i / 255.0, 4) * 255.0, 0, 255)
    return gammalookUpTable

def RemoveHighlight(gammalookUpTable, img):
    img = cv2.bitwise_not(img)
    return cv2.LUT(img, gammalookUpTable)

def DecodeImage(model, hog, imgGS):
    thresh1 = cv2.threshold(imgGS, 40, 255, cv2.THRESH_BINARY)[1]
    thresh1 = cv2.bitwise_not(thresh1)

    tmp = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    tmp = tmp[0] if len(tmp) == 2 else tmp[1]
    tmp = sorted(tmp, key = lambda x: cv2.boundingRect(x)[0])

    hog_descriptors = []
    for c in tmp:        
        x, y, w, h = cv2.boundingRect(c)
        if w > 8 and w < 30 and h > 16:
            imgchar = imgGS[y:y+h, x:x+w]
            h, w = imgchar.shape[:2]
            if h <= 25 and w <= 25:
                imgchar = cv2.copyMakeBorder(imgchar, 0, 25-h, 0, 25-w, cv2.BORDER_CONSTANT, 0)
                hog_descriptors.append(hog.compute(imgchar))
    
    if len(hog_descriptors) < 3:
        return ""
    hog_descriptors = np.squeeze(hog_descriptors)
    predictions = model.predict(hog_descriptors)
    text = predictions[1].ravel()
    return f"{text[0]:2.0f}-{text[1]:2.0f}-{text[2]:2.0f}"

def checkFiles():
    whitecode = "whitecode.png"
    whitepin = "whitepin.png"
    model = "code_recognition_model.yml"
    flag = False

    if not osPath.exists(whitecode):
        flag = True
        print("Missing", whitecode)
        print("Check if they are in the same directory as this file")
    if not osPath.exists(model):
        flag = True
        print("Missing", model)
        print("Check if they are in the same directory as this file")
    if not osPath.exists(whitepin):
        flag = True
        print("Missing", whitepin)
        print("Check if they are in the same directory as this file")
    return flag

def atmCracker():
    model = cv2.ml.SVM_load("code_recognition_model.yml")
    hog = get_hog()
    gammatable = GetGammaTable()

    active = False
    justPressed = False
    justClicked = False

    ATMWantedCodeLocation = { # code location for 1920x1080 fullscreened game (rough code block)
        "top": 288,
        "left": 1052,
        "width": 71,
        "height": 22
        }

    ATMWantedCodeLocation2 = { # code location for 1920x1080 fullscreened game (full code block)
        "top": 281,
        "left": 965,
        "width": 238,
        "height": 36
        }

    ATMWantedCodeLocation3 = { # code location for 1920x1080 fullscreened game (smaller code block)
        "top": 282,
        "left": 980,
        "width": 220,
        "height": 34
        }

    ATMCurrentCodeLocation = { # list of codes location for 1920x1080 fullscreened game
        "left": 580,
        "top": 374,
        "width": 758,
        "height": 430
        }

    white = cv2.imread("whitecode.png")
    white = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
    
    clear()
    print("Hold \"Alt + Q\" to start cracking")
    print("Hold \"End\" to return to menu")
    while True:
        if kb.is_pressed("alt+q"): # calling this rapidly slows keyboard responses. so when the cracker is off it only runs a "q" check every 1 second rather than every possible time
            if not justPressed:
                justPressed = True
                active = not active
                if active == True:
                    print("ATM Cracker on")
                    print("Don't move your mouse!")
                    print("If so hover over ATM gui")
                    prepareMouse()
                else:
                    print("ATM Cracker off")
        elif kb.is_pressed("end"):
            if active == False:
                if not justPressed:
                    #print("break")
                    clear()
                    return
        else:
            justPressed = False

        if active:
            ss = mss.mss()

            codeC = np.array(ss.grab(ATMWantedCodeLocation3))
            codesC = np.array(ss.grab(ATMCurrentCodeLocation))
                        
            codeGS = cv2.cvtColor(codeC, cv2.COLOR_BGR2GRAY)
            codesGS = cv2.cvtColor(codesC, cv2.COLOR_BGR2GRAY)


            codetext = DecodeImage(model, hog, codeGS)
            
            codes = cv2.threshold(codesGS, 20, 255, cv2.THRESH_BINARY)[1] # converts to black and white
            result = cv2.matchTemplate(codes, white, cv2.TM_CCORR_NORMED) # finds highlighted area

            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= 0.8:
                w, h = white.shape[::-1]
                top_left = max_loc
                
                codestextImage = codesGS[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w] # shrinks code list image into the highlighted section aka the current code

                codestextImage = RemoveHighlight(gammatable, codestextImage)
                codestext = DecodeImage(model, hog, codestextImage) 

                # Problem - If multiple of the same code is on the codes list, it will choose the first one it spots. Most of the time, there isn't a high chance of that happening.
                #           Luckly it seems like there can only be 2 of the same code, possibly a server side check. Only fix is colour checking.
                if codetext != "" and codestext != "":
                    print(codetext, codestext)

                if codetext == codestext and codetext != "":
                    if not justClicked:
                        justClicked = True
                        print("-=========================-")
                        print("-=========================-")
                        print("")
                        print("       FOUND MATCH")
                        print("")
                        print("required code =", codetext)
                        print("highlighted code =", codestext)
                        print("")
                        print("-=========================-")
                        print("-=========================-")

                        clickLeft()
                else:
                    justClicked = False
            else:
                print("cant find highlighted area, val =", max_val)
                print("x =", max_loc[0])
        if active == False:
            sleep(1)

def autoDriller():
    clear()
    print("Hold \"Alt + Q\" to start drilling")
    print("Hold \"End\" to return to menu")

    active = False
    justPressed = False
    waitTime = 0.001

    while True:
        if kb.is_pressed("alt+q"): # calling this rapidly slows keyboard responses. so when the cracker is off it only runs a "q" check every 1 second rather than every possible time
            if not justPressed:
                justPressed = True
                active = not active
                if active == True:
                    print("Auto Driller on")
                    print("Don't move your mouse!")
                    print("If so hover over robbery gui")
                    prepareMouse()
                else:
                    print("Auto Driller off")
        elif kb.is_pressed("end"):
            if active == False:
                if not justPressed:
                   # print("break")
                    clear()
                    #for i in range(0, 100):
                    #    print("")
                    return    
        else:
            justPressed = False

        if active:
            mouse.press(button="left")
            sleep(waitTime)
            mouse.release(button="left")
            sleep(waitTime)
        else:
            sleep(1)

def lockPicker():
    active = False
    justPressed = False
    justClicked = False

    gobals = [536, 1, 9] # top, width height
    pinLocation1 = { # pin1 location for 1920x1080 fullscreened game
        "top": gobals[0],
        "left": 772, # 748 old
        "width": gobals[1],
        "height": gobals[2]
        }

    pinLocation2 = { # pin2 location for 1920x1080 fullscreened game
        "top": gobals[0],
        "left": 772 + 75, # 823
        "width": gobals[1],
        "height": gobals[2]
        }

    pinLocation3 = { # pin3 location for 1920x1080 fullscreened game
        "top": gobals[0],
        "left": 772 + (75 * 2),
        "width": gobals[1],
        "height": gobals[2]
        }

    pinLocation4 = { # pin4 location for 1920x1080 fullscreened game
        "top": gobals[0],
        "left": 772 + (75 * 3),
        "width": gobals[1],
        "height": gobals[2]
        }

    pinLocation5 = { # pin5 location for 1920x1080 fullscreened game
        "top": gobals[0],
        "left": 772 + (75 * 4),
        "width": gobals[1],
        "height": gobals[2]
        }

    pinLocation6 = { # pin6 location for 1920x1080 fullscreened game
        "top": gobals[0],
        "left": 772 + (75 * 5),
        "width": gobals[1],
        "height": gobals[2]
        }

    pinLocations = [pinLocation1, pinLocation2, pinLocation3, pinLocation4, pinLocation5, pinLocation6]
    currentPin = 0 # add 1 for current

    white = cv2.imread("whitepin.png")
    white = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
    missed = 0

    clear()
    print("Hold \"Alt + Q\" to start lockpicking")
    print("Hold \"End\" to return to menu")
    while True:
        if kb.is_pressed("alt+q"): # calling this rapidly slows keyboard responses. so when the cracker is off it only runs a "q" check every 1 second rather than every possible time
            if not justPressed:
                justPressed = True
                active = not active
                if active == True:
                    print("Auto Lockpicker on")
                    print("Don't move your mouse!")
                    print("If so hover over the lockpicking gui")

                    prepareMouse()
                    currentPin = 0
                    missed = 0
                else:
                    print("Auto Lockpicker off")
        elif kb.is_pressed("end"):
            if active == False:
                if not justPressed:
                    clear()
                    return    
        else:
            justPressed = False

        if active:
            if currentPin >= 6:
                print("Picked all pins")
                print("Auto Lockpicker off")
                active = False
            else:
                ss = mss.mss()

                pinC = np.array(ss.grab(pinLocations[currentPin]))

                pinGS = cv2.cvtColor(pinC, cv2.COLOR_BGR2GRAY)
                
                #cv2.imwrite("pin_true.png", pinGS)
                
                pin = cv2.threshold(pinGS, 120, 255, cv2.THRESH_BINARY)[1] # converts to black and white
                #print(pin)
                #print(pin.sum())
                #cv2.imshow("img", pin)
                #cv2.waitKey()
                #result = cv2.matchTemplate(pin, white, cv2.TM_CCORR_NORMED)

                #_, max_val, _, _ = cv2.minMaxLoc(result)
                #print(max_val)

                if pin.sum() == 2295: # 2295 is 255*9 aka when all 9 pixels are white
                    if not justClicked and missed == 10:
                        clickLeft()
                        justClicked = True
                        print("Found pin " + str((currentPin + 1)) + "!")
                        currentPin = currentPin + 1
                        
                        print("clicked")
                    else:
                        missed = missed + 1

                else:
                    justClicked = False
        else:
            sleep(1)

def autoSafe():
    active = False
    justPressed = False
    justClicked = False

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

    while True:
        if kb.is_pressed("alt+q"): # calling this rapidly slows keyboard responses. so when the cracker is off it only runs a "q" check every 1 second rather than every possible time
            if not justPressed:
                justPressed = True
                active = not active
                if active == True:
                    print("Auto Lockpicker on")
                    print("Don't move your mouse!")
                    print("If so hover over the lockpicking gui")

                    prepareMouse()
                else:
                    print("Auto Lockpicker off")
        elif kb.is_pressed("end"):
            if active == False:
                if not justPressed:
                    clear()
                    return    
        else:
            justPressed = False

        if active:
            ss = mss.mss()

            tarNumC = np.array(ss.grab(targetNumberLoc))
            safeNumC = np.array(ss.grab(safeCodeLoc))

            tarNumGS = cv2.cvtColor(tarNumC, cv2.COLOR_BGR2GRAY)
            safeNumGS = cv2.cvtColor(safeNumC, cv2.COLOR_BGR2GRAY)
        else:
            sleep(1)

def quickToolBuy():
    prepareMouse()
    clickLeft()
    buyTool(189, 438)
    buyTool(242, 481)
    buyTool(315, 566)
    moveMouse(1863, 82)
    clickLeft()



def main():
    clear()
    if checkFiles(): # exits the program if theres missing files
        return

    print("Best ran through command prompt")
    while True:
        print("Roblox ER:LC Robbery Cheats - Version 1.0-beta")
        print("")
        print("Choose a cheat:")
        print("A = ATM Cracker")
        print("B = Auto Driller")
        print("C = Auto Lockpicker")
        print("T = Quick Buy Tools")
        print("E = Close Program")
        
        userIn = input()

        if userIn == "a" or userIn == "A":
            atmCracker()
        elif userIn == "b" or userIn == "B":
            autoDriller()
        elif userIn == "c" or userIn == "C":
            lockPicker()
        elif userIn == "d" or userIn == "D":
            autoSafe()
        elif userIn == "e" or userIn == "E":
            print("Stopping...")
            return
        elif userIn == "t" or userIn == "T":
            quickToolBuy()
        else:
            print("Invalid selection...")



if __name__ == '__main__':
    main()
