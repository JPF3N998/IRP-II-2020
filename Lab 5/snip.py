from matplotlib import pyplot as plt
import os
from os import listdir, getcwd
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
    plt.figure(figsize=(5, 5))
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

    resumedHist = []

    for multiplo in range(1, 5):
        pixelBlock = 0
        for i in range(height * (multiplo - 1), height * multiplo):
            pixelBlock += histH[i]
            
        resumedHist.append(pixelBlock)

    for multiplo in range(1, 5):
        pixelBlock = 0
        for i in range(width * (multiplo - 1), width * multiplo):
            pixelBlock += histV[i]
            
        resumedHist.append(pixelBlock)

    return resumedHist

def readAndCropImage(img):

    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)#se convierte a una imagen binaria 0 | 255
    dataset = cropNumbers(img)
    writeDataset(dataset)

def getHistsProm(hists):

    prom = [0, 0, 0, 0, 0, 0, 0, 0]

    for hist in hists:
        prom = np.add(prom, hist)

    prom = np.true_divide(prom, len(hists))

    return prom.tolist()

def evalNumber(number, hists):

    histNumber = getHist4x4(number)
    print(histNumber)

    distances = []

    for hist in hists:        
        dist = np.subtract(histNumber, hist)
        dist = np.abs(dist)
        distances.append(np.sum(dist))

    return distances.index(min(distances))

def readNumbers(readFolder):
    cwd = os.getcwd()
    pathTrain = cwd + "/" + readFolder
    trainFiles = os.listdir(path=pathTrain)

    hists = []
    for imageName in trainFiles:
        number = cv.imread(readFolder + imageName, 0)
        hists.append(getHist4x4(number))

    return hists



pivoteColor = 200
resizeDim = (50, 150)#imagenes de 100 x 100 para los números individuales

"""
promHists = []
for i in range(0, 10):
    promHists.append(getHistsProm(readNumbers(str(i) + "/")))

print(promHists)
"""

asd = [[559.1166666666667, 597.9833333333333, 549.6666666666666, 619.9333333333333, 660.2666666666667, 517.2833333333333, 475.5, 617.4333333333333],
        [752.4754098360655, 1361.5901639344263, 1338.8688524590164, 905.6065573770492, 451.62295081967216, 1254.983606557377, 1579.950819672131, 1048.2622950819673],
        [610.2622950819672, 540.8360655737705, 687.8524590163935, 905.9180327868852, 574.639344262295, 817.7868852459017, 929.7213114754098, 411.1639344262295],
        [761.4590163934427, 716.3934426229508, 710.360655737705, 833.983606557377, 396.1639344262295, 757.688524590164, 948.7213114754098, 881.1147540983607],
        [534.2295081967213, 1130.5573770491803, 840.344262295082, 470.5081967213115, 666.7213114754098, 528.360655737705, 683.9016393442623, 1045.1311475409836],
        [808.8196721311475, 905.9180327868852, 869.6229508196722, 790.5409836065573, 730.4918032786885, 1038.4426229508197, 860.0819672131148, 723.8196721311475],
        [621.0655737704918, 702.5409836065573, 1261.5245901639344, 941.672131147541, 896.4426229508197, 1022.0655737704918, 866.9508196721312, 713.5409836065573],
        [986.655737704918, 623.5409836065573, 823.6065573770492, 380.1475409836066, 376.2295081967213, 710.311475409836, 773.7377049180328, 904.2459016393443],
        [995.0491803278688, 1177.2459016393443, 1189.4918032786886, 988.8852459016393, 944.5573770491803, 1162.2295081967213, 1179.1147540983607, 1031.295081967213],
        [920.688524590164, 1126.7377049180327, 898.5573770491803, 548.1311475409836, 682.360655737705, 866.983606557377, 811.7049180327868, 1090.9016393442623]]

numero = cv.imread("numero.jpg", 0)
plotImage(numero)
numero = cv.resize(numero, resizeDim, interpolation = cv.INTER_AREA)
print("El número es:", evalNumber(numero, asd))




"""
readFolder = "Jose/"
saveFolder = "Jose/"

numberToRead = "9"

readFolder = readFolder + numberToRead 
saveFolder = saveFolder + numberToRead + "/"

readAndCropImage(cv.imread(readFolder + "/" + numberToRead + ".jpg", 0))
"""

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