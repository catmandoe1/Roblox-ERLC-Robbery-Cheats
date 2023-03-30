from asyncio.windows_events import NULL
from pickle import NONE
import cv2
import numpy as np



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
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR



def checkverticleimage():
    imgcode = cv2.imread("vertimg.png")

    mychars = ["B","F","8","B","B","F","4","7","3","F","7","D","D","C","6","E","F","F","3","F","F","A","A","5","E","E","D","E","B","E","2","8","3","B","B","A","C","2","2","5","B","E","4","0","4","8","0","A","D","E","9","E","9","D","4","A","A","4","B","B","5","C","7","C","C","D","7","0","A","A","A","A","5","4","7","2","4","7","5","8","A","D","2","D","0","7","C","6","0","A","C","0","C","0","5","8","F","3","7","A","F","3","2","8","D","0","D","F","2","D","E","2","7","A","3","0","D","A","3","C","E","E","F","6","7","A","A","B","2","E","9","0","E","7","1","9","B","3","A","B","A","A","5","C","8","2","2","B","0","E","2","8","5","5","E","F","A","6","V","8","C","E","D","F","3","4","D","7","5","E","2","B","9","5","7","8","D","C","2","8","0","E","B","4","F","A","4","C","A","F","3","5","E","6","9","A","C","D","D","D","9","6","6","F","E","A","2","C","F","D","5","F","8","8","B","A","4","E","F","5","C","D","4","8","0","F","F","E","3","1","6","2","B","9","1","A","A","3","1","C","8","B","F","1","1","B","C","A","C","A","6","3","3","6","5"]

    #imgcode = cv2.copyMakeBorder(imgcode, 10, 10, 10, 10, cv2.BORDER_CONSTANT, 0)
    #imgcode = cv2.cvtColor(imgcode, cv2.COLOR_BGR2GRAY)
    #imgcode = cv2.threshold(imgcode, 20, 255, cv2.THRESH_BINARY)[1]
    #imgcode = cv2.bitwise_not(imgcode)

    pos =0
    for i in range(0,6375,25):
        imgchar = imgcode[i:i+25, 0:25]   
        imgchar = cv2.copyMakeBorder(imgchar, 10, 100, 200, 10, cv2.BORDER_CONSTANT, 0) 
        myc = mychars[pos]
        cv2.imshow(myc,imgchar)
        cv2.waitKey()
        cv2.destroyAllWindows()
        pos=pos+1


def testdecode():
        
    model = cv2.ml.SVM_load('roblox_model_vert.yml')
    hog = get_hog()

    #1 Define path of document(s)
    #imgcodeOrig = cv2.imread("sct-{top}x{left}_{width}x{height}.png")
    #imgcodeOrig = cv2.imread("atmcorrectc.png")
    #imgcodeOrig = cv2.imread("lots.png")
    #imgcodeOrig = cv2.imread("tomatchcode.png")
    imgcodeOrig = cv2.imread("ss1.png")
    

    cv2.imshow('image',imgcodeOrig)
    cv2.waitKey(0)  

    imgcode = cv2.cvtColor(imgcodeOrig, cv2.COLOR_BGR2GRAY)

    #x,y,w,h = cv2.boundingRect(imgcode)
    #imgcode = imgcode[y:y+h, x:x+w]

    #imgcode = cv2.copyMakeBorder(imgcode, 10, 10, 10, 10, cv2.BORDER_CONSTANT, 0)
    thresh1 = cv2.threshold(imgcode, 20, 255, cv2.THRESH_BINARY)[1]
    thresh1 = cv2.bitwise_not(thresh1)

    #cv2.imshow('image',thresh1)
    #cv2.waitKey(0)  

    #ret,thresh1 = cv2.threshold(imgcode,20,255,cv2.THRESH_BINARY)
    #thresh1 = cv2.threshold(imgcode, 0, 100, cv2.THRESH_BINARY)[1]
    tmp = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #tmp = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tmp = tmp[0] if len(tmp) == 2 else tmp[1]
    tmp = sorted(tmp, key = lambda x: cv2.boundingRect(x)[0])
    i = 0
    hog_descriptors = []
    ri = imgcodeOrig.copy()        
    vertimg = NONE
    for c in tmp:        

        x, y, w, h = cv2.boundingRect(c)
        
        
        if w > 8 and w < 30 and h > 18:
            i = i+1
            print(i)
            print(x, y, w, h)
            
            cv2.rectangle(ri, (x,y),(x+w, y+h),(0,255,12),1)
            imgchar = imgcode[y:y+h, x:x+w]

            h, w = imgchar.shape[:2]
            imgchar = cv2.copyMakeBorder(imgchar, 0, 25-h, 0, 25-w, cv2.BORDER_CONSTANT, 0)
            #print(w,h)
            #imgchar = cv2.resize(imgchar, (25,25), interpolation = cv2.INTER_AREA)
            #h, w = imgchar.shape[:2]
            #print(w,h)
            #fname = f"out-{i}.png"
            #cv2.imwrite(fname, imgchar)

            if vertimg is NONE:
                vertimg = imgchar
            else:
                vertimg = np.concatenate((vertimg, imgchar), axis=0)


            #cv2.imshow('image',imgchar)
            #cv2.waitKey(0)  
            hog_descriptors.append(hog.compute(imgchar))
            #print("append hog")
        
    #cv2.imwrite("vertimg.png", vertimg)
    print("loop end",i)

    hog_descriptors = np.squeeze(hog_descriptors)
    predictions = model.predict(hog_descriptors)
    predictions = predictions[1].ravel()
    print(predictions)

    cv2.imshow('image',vertimg)
    cv2.imshow('image',ri)
    cv2.waitKey(0)  

 
def GetGammaTable():
    gammalookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        gammalookUpTable[0,i] = np.clip(pow(i / 255.0, 3) * 255.0, 0, 255)
    return gammalookUpTable

def RemoveHighlight(gammalookUpTable, img):
    img = cv2.bitwise_not(img)
    return cv2.LUT(img, gammalookUpTable)

def DecodeImage(model, hog, imgGS):
    #print("decode")    
    #imgcode = cv2.cvtColor(imgcodeOrig, cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.threshold(imgGS, 20, 255, cv2.THRESH_BINARY)[1]
    thresh1 = cv2.bitwise_not(thresh1)
    #cv2.imwrite("xxss3thresh.png",thresh1)

    tmp = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    tmp = tmp[0] if len(tmp) == 2 else tmp[1]
    tmp = sorted(tmp, key = lambda x: cv2.boundingRect(x)[0])

    hog_descriptors = []
    for c in tmp:        
        x, y, w, h = cv2.boundingRect(c)
        #print(w,h)
        if w > 8 and w < 30 and h > 16:
            #print("adding")
            imgchar = imgGS[y:y+h, x:x+w]
            h, w = imgchar.shape[:2]
            imgchar = cv2.copyMakeBorder(imgchar, 0, 25-h, 0, 25-w, cv2.BORDER_CONSTANT, 0)
            hog_descriptors.append(hog.compute(imgchar))
        
    hog_descriptors = np.squeeze(hog_descriptors)
    predictions = model.predict(hog_descriptors)
    return  predictions[1].ravel()


def DecodeImageTest():
        
    model = cv2.ml.SVM_load('roblox_model_vert.yml')
    hog = get_hog()
    gammatable = GetGammaTable()

    #1 Define path of document(s)
    #imgcodeOrig = cv2.imread("sct-{top}x{left}_{width}x{height}.png")
    #imgcodeOrig = cv2.imread("atmcorrectc.png")
    #imgcodeOrig = cv2.imread("lots.png")
    #imgcodeOrig = cv2.imread("tomatchcode.png")


    imgcodeOrig = cv2.imread("ss1.png")
    imgcodeOrig = cv2.cvtColor(imgcodeOrig, cv2.COLOR_BGR2GRAY)
    result = DecodeImage(model, hog, imgcodeOrig)
    print(result)
    cv2.imshow('image',imgcodeOrig)
    cv2.waitKey() 
    
    
    imgcodeOrig = cv2.imread("ss2.png")
    imgcodeOrig = cv2.cvtColor(imgcodeOrig, cv2.COLOR_BGR2GRAY)
    imgcodeOrig = RemoveHighlight(gammatable, imgcodeOrig)
    result = DecodeImage(model, hog, imgcodeOrig)
    print(result)
    cv2.imshow('image',imgcodeOrig)
    cv2.waitKey() 

    imgcodeOrig = cv2.imread("ss3.png")
    imgcodeOrig = cv2.cvtColor(imgcodeOrig, cv2.COLOR_BGR2GRAY)
    imgcodeOrig = RemoveHighlight(gammatable, imgcodeOrig)
    cv2.imwrite("xxss3.png",imgcodeOrig)
    result = DecodeImage(model, hog, imgcodeOrig)
    print(result)
    cv2.imshow('image',imgcodeOrig)
    cv2.waitKey()  

if __name__ == '__main__':
    DecodeImageTest()
