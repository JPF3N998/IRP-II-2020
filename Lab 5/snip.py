from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

def snipHorizontal(img):
    crops = []
    height = img.shape[0]
    y = 0

    while(y < height):
        if(not freeHorizontal(img, y)):
            upperSnipY = upperSnip(img, y)
            bottomSnipY = bottomSnip(img, y)
            
            if(upperSnipY == -1 or bottomSnipY == -1):
                print(y)
                print("Error")
            else:
                crops.append((upperSnipY, bottomSnipY))
                y = bottomSnipY
            
                
        y += 1

    return crops

def freeHorizontal(img, y):
    width = img.shape[1]

    for x in range(0, width):
        if(img[y][x] <= pivoteColor):
            return False
        
    return True

def upperSnip(img, firstY):

    for y in range(0, firstY):
        y = firstY - y
        if(freeHorizontal(img, y)):
            return y

    return -1#no debería ocurrir

def bottomSnip(img, firstY):
    height = img.shape[0]

    for y in range(firstY, height):
        if(freeHorizontal(img, y)):
            return y

    return -1

def snipVertical(img):
    crops = []
    width = img.shape[1]

    x = 0

    while(x < width):
        if(not freeVertical(img, x)):
            leftSnipX = leftSnip(img, x)
            rightSnipX = rightSnip(img, x)
            
            if(leftSnipX == -1 or rightSnipX == -1):
                print(y)
                print("Error")
            else:
                crops.append((leftSnipX, rightSnipX))
                x = rightSnipX
                
        x += 1
    return crops

def freeVertical(img, x):
    height = img.shape[0]

    for y in range(0, height):
        if(img[y][x] <= pivoteColor):
            return False
        
    return True

def leftSnip(img, firstX):

    for x in range(0, firstX):
        x = firstX - x
        if(freeVertical(img, x)):
            return x

    return -1#no debería ocurrir

def rightSnip(img, firstX):
    width = img.shape[1]

    for x in range(firstX, width):
        if(freeVertical(img, x)):
            return x

    return -1

def cropFila(img, crop):
    width = img.shape[1]
    
    return img[crop[0]: crop[1], 0: width - 1]

def cropColumn(img, crop):
    height = img.shape[0]
    
    return img[0: height - 1, crop[0]: crop[1]]

def cropNumbers(img):
    dataset = []

    crops = snipHorizontal(img)#cortes por fila

    filas = []
    for crop in crops:
        filas.append(cropFila(img, crop))

    for fila in filas:
        crops = snipVertical(fila)#cortes de cada fila por número
        
        for crop in crops:
            dataset.append(cropColumn(fila, crop))#cortes por número
            
    return dataset

def writeDataset(dataset):
    i = 1
    for image in dataset:
        cv.imwrite(str(i) + ".jpg", image)
        i += 1

def plotImage(img):
    plt.figure(figsize=(5, 5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap = 'gray')
    plt.title('Original')
    plt.show()

def getHorizontalHist(number):
    width = number.shape[1]
    height = number.shape[0]
    
    hist = []
    for y in range(height):
        pixelCount = 0
        for x in range(width):
            if(number[y][x] <= pivoteColor):
                pixelCount += 1
                
        hist.append(pixelCount)

    return hist

def getVerticalHist(number):
    width = number.shape[1]
    height = number.shape[0]
    
    hist = []
    for x in range(width):
        pixelCount = 0
        for y in range(height):    
            if(number[y][x] <= pivoteColor):
                pixelCount += 1
                
        hist.append(pixelCount)

    return hist

def getHist4x4(number):
    width = int(number.shape[1] / 4)
    height = int(number.shape[0] / 4)

    histH = getHorizontalHist(number)
    histV = getVerticalHist(number)

    resumedH = []
    resumedV = []

    for multiplo in range(1, 5):
        pixelBlock = 0
        for i in range(height * (multiplo - 1), height * multiplo):
            pixelBlock += histH[i]
            
        resumedH.append(pixelBlock)

    for multiplo in range(1, 5):
        pixelBlock = 0
        for i in range(width * (multiplo - 1), width * multiplo):
            pixelBlock += histV[i]
            
        resumedV.append(pixelBlock)

    return (resumedH, resumedV)
    


img = cv.imread("original4.jpg", 0)

ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)#se convierte a una imagen binaria 0 | 255

#plotImage(img)

pivoteColor = 200


#dataset = cropNumbers(img)
#writeDataset(dataset)


number = cv.imread("9.jpg", 0)
hists = getHist4x4(number)

print(hists)

plt.figure(figsize=(8, 5))
plt.subplot(1,1,1)
plt.subplot(131)
plt.bar([1, 2, 3, 4], np.array(hists[0]))
plt.subplot(132)
plt.bar([1, 2, 3, 4], np.array(hists[1]))
plt.title('Original')
plt.show()
