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
        cv.imwrite(saveFolder + "0" + str(i) + ".jpg", cv.resize(image, resizeDim, interpolation = cv.INTER_AREA))
        i += 1

def plotImage(img):
    plt.figure(figsize=(10, 8))
    plt.subplot(1,1,1)
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

def readAndCropImage(img):

    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)#se convierte a una imagen binaria 0 | 255
    plotImage(img)
    dataset = cropNumbers(img)
    writeDataset(dataset)

def evalNumber(number, hists = []):

    hists = [([551, 379, 368, 454], [527, 323, 315, 513]), ([649, 940, 818, 377], [141, 927, 1017, 625]), ([112, 256, 256, 625], [214, 400, 412, 215]), ([167, 380, 383, 494], [156, 367, 496, 399]), ([248, 639, 479, 176], [423, 151, 212, 743]), ([235, 439, 363, 436], [223, 447, 381, 406]), ([282, 366, 786, 706], [650, 535, 434, 483]), ([574, 347, 502, 188], [124, 353, 608, 503]), ([580, 687, 508, 573], [402, 635, 587, 709]), ([533, 624, 435, 229], [316, 296, 367, 797])]

    histNumber = getHist4x4(number)

    distances = []

    for hist in hists:
        dist = 0
        for i in range(len(hist)):
            dist += abs(hist[0][i] - histNumber[0][i])
            dist += abs(hist[1][i] - histNumber[1][i])

        distances.append(dist)

    return distances.index(min(distances))

pivoteColor = 200
resizeDim = (50, 150)#imagenes de 100 x 100 para los números individuales

readFolder = "Jose/"
saveFolder = "Jose/"

numberToRead = "9"

readFolder = readFolder + numberToRead 
saveFolder = saveFolder + numberToRead + "/"

readAndCropImage(cv.imread(readFolder + "/" + numberToRead + ".jpg", 0))

#for i in range(0, 10):
    #print("Número real:", i, "predicción: ", evalNumber(cv.imread(str(i) + ".jpg", 0)))


"""
number = cv.imread("0.jpg", 0)
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

a = []
for i in range(0, 10):
    a.append(getHist4x4(cv.imread(str(i) + ".jpg", 0)))
print(a)
print()

#histogramas para el dataset original resized to 150x50
A = [([533, 415, 428, 402], [516, 307, 270, 609]), ([518, 555, 337, 193], [111, 251, 790, 433]), ([93, 176, 224, 568], [201, 389, 351, 112]), ([300, 228, 413, 496], [136, 411, 509, 380]), ([385, 977, 383, 161], [522, 180, 381, 793]), ([572, 398, 212, 470], [246, 496, 435, 462]), ([299, 384, 879, 949], [806, 561, 537, 570]), ([784, 348, 413, 341], [137, 481, 783, 469]), ([512, 724, 440, 610], [369, 524, 516, 839]), ([588, 752, 409, 297], [260, 388, 566, 791])]

#histogramas para el dataset original5 resized to 150x50
B = [([344, 169, 210, 271], [332, 168, 174, 244]), ([690, 784, 706, 609], [151, 1459, 965, 210]), ([146, 187, 182, 429], [109, 184, 261, 359]), ([201, 226, 177, 403], [119, 227, 347, 258]), ([162, 280, 207, 106], [210, 60, 61, 410]), ([92, 98, 237, 248], [66, 275, 160, 177]), ([69, 180, 462, 520], [369, 340, 214, 286]), ([225, 179, 479, 114], [80, 218, 589, 96]), ([505, 506, 330, 246], [342, 385, 363, 465]), ([326, 420, 161, 111], [143, 105, 124, 589])]

#histogramas para el dataset original6 resized to 150x50
C = [([776, 553, 466, 688], [732, 493, 500, 687]), ([740, 1480, 1410, 330], [160, 1072, 1296, 1232]), ([96, 405, 362, 879], [331, 627, 623, 175]), ([0, 685, 560, 582], [212, 464, 631, 558]), ([197, 661, 847, 261], [538, 212, 195, 1025]), ([40, 821, 639, 589], [356, 569, 548, 578]), ([479, 533, 1018, 650], [774, 704, 551, 593]), ([714, 513, 613, 108], [154, 359, 452, 945]), ([723, 831, 753, 862], [496, 995, 883, 823]), ([684, 700, 736, 280], [544, 394, 410, 1010])]

#promedio de los 3 histogramas
[([551, 379, 368, 454], [527, 323, 315, 513]), ([649, 940, 818, 377], [141, 927, 1017, 625]), ([112, 256, 256, 625], [214, 400, 412, 215]), ([167, 380, 383, 494], [156, 367, 496, 399]), ([248, 639, 479, 176], [423, 151, 212, 743]), ([235, 439, 363, 436], [223, 447, 381, 406]), ([282, 366, 786, 706], [650, 535, 434, 483]), ([574, 347, 502, 188], [124, 353, 608, 503]), ([580, 687, 508, 573], [402, 635, 587, 709]), ([533, 624, 435, 229], [316, 296, 367, 797])]

D = []
#saca el promedio de los arreglos
for i in range(10):
    innerH = []
    innerV = []
    for j in range(4):
        innerH.append(round((A[i][0][j] + B[i][0][j] + C[i][0][j]) / 3))
        innerV.append(round((A[i][1][j] + B[i][1][j] + C[i][1][j]) / 3))
    D.append((innerH, innerV))
    
print(D)

"""