from matplotlib import pyplot as plt
import os
from os import listdir, getcwd
import numpy as np
import cv2 as cv
import numpy as np

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
            numero = cropColumn(fila, crop)
            numero = crop_image(numero)
            dataset.append(numero)#cortes por número
            
    return dataset

def writeDataset(dataset, saveFolder):
    i = 90
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

def crop_image(img, tol = 80):
#recuperado de
#https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934
    # img is 2D image data
    # tol  is tolerance
    mask = img < tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def getHist4x4(number, pixelWindow):
    width = number.shape[1]
    height = number.shape[0]

    if(width % pixelWindow != 0 or height % pixelWindow != 0):
        newDim = (int(width / pixelWindow) * pixelWindow, int(height / pixelWindow) * pixelWindow)
        number = cv.resize(number, newDim, interpolation = cv.INTER_AREA)

    hist = getHorizontalHist(number) + getVerticalHist(number)
    resumedHist = []

    for i in range(0, len(hist), pixelWindow):
        pixelCount = 0
        for j in range(i, i + pixelWindow):
            pixelCount += hist[j]

        resumedHist.append(pixelCount)
    
    return resumedHist

def readAndCropImage(img, saveFolder):

    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)#se convierte a una imagen binaria 0 | 255
    dataset = cropNumbers(img)
    writeDataset(dataset, saveFolder)

def getHistsProm(hists):

    prom = [0] * len(hists[0])

    for hist in hists:
        prom = np.add(prom, hist)

    prom = np.true_divide(prom, len(hists))

    return np.around(prom,decimals=4).tolist()

def evalNumber(number, hists):
    #pre procesamiento
    ret, number = cv.threshold(number, 127, 255, cv.THRESH_BINARY)#se convierte a una imagen binaria 0 | 255
    number = crop_image(number)
    number = cv.resize(number, resizeDim, interpolation = cv.INTER_AREA)
    
    histNumber = getHist4x4(number, pixelWindowForHist)

    distances = []

    for hist in hists:
        dist = np.linalg.norm(np.array(histNumber) - np.array(hist))
        distances.append(np.sum(dist))

    return distances.index(min(distances))

def readNumbers(readFolder):
    cwd = os.getcwd()
    pathTrain = cwd + "/" + readFolder
    trainFiles = os.listdir(path=pathTrain)

    hists = []
    for imageName in trainFiles:
        number = cv.imread(readFolder + imageName, 0)
        hists.append(getHist4x4(number, pixelWindowForHist))

    return hists

def generate_sub_dataset(readFolder, saveFolder, numberToRead):
    readFolder = readFolder + numberToRead 
    saveFolder = saveFolder + numberToRead + "/"
    readAndCropImage(cv.imread(readFolder + "/" + numberToRead + ".png", 0), saveFolder)

pivoteColor = 200
resizeDim = (50, 150)#imagenes de 100 x 100 para los números individuales
pixelWindowForHist = 4

for i in range(0, 10):
    generate_sub_dataset("Feng/", "Feng/", str(i))

#promHists = []
#for i in range(0, 10):
#    promHists.append(getHistsProm(readNumbers(str(i) + "/")))

#print(promHists)

#asd = [[3.9832, 14.0756, 25.437, 37.9748, 48.8571, 58.7311, 64.2689, 67.4034, 69.3361, 69.6303, 65.4034, 59.2269, 54.4874, 50.3529, 47.7647, 45.0, 43.7983, 42.5042, 41.6975, 41.0504, 41.5714, 42.3361, 43.1092, 45.0, 48.1176, 51.2269, 55.8319, 61.2941, 66.9328, 71.7227, 69.0672, 61.0504, 53.5882, 44.5294, 35.5798, 28.0084, 17.9328, 137.5042, 232.7983, 169.6639, 139.9748, 121.3193, 115.3866, 118.2689, 128.0, 129.7479, 142.5378, 189.8487, 162.8319], [7.1333, 23.075, 39.2833, 51.1167, 62.65, 74.5667, 84.5667, 91.6667, 99.3333, 103.8833, 108.7417, 111.7833, 112.9, 113.0833, 112.8, 113.4833, 112.7, 112.025, 111.5917, 111.3417, 110.0667, 110.1417, 109.7667, 109.875, 109.1167, 108.5, 109.125, 106.875, 104.25, 98.8333, 89.9583, 79.5583, 67.975, 58.1333, 49.3333, 38.075, 24.8167, 12.425, 66.1, 164.0833, 234.2417, 324.6667, 380.6417, 432.1417, 437.9167, 418.2417, 376.8167, 243.625, 151.225], [4.475, 14.9667, 24.6417, 33.05, 41.8833, 48.7167, 51.4417, 52.8333, 51.0083, 46.1417, 40.2667, 35.2417, 30.6833, 28.1833, 26.1417, 24.2167, 23.9583, 23.4, 23.7083, 24.6917, 27.7833, 31.4917, 35.9917, 41.55, 46.8667, 53.4, 58.6083, 62.4667, 67.3917, 73.1417, 79.3417, 83.0083, 
#80.45, 68.5583, 54.0667, 37.7083, 22.0167, 60.0167, 135.3583, 153.05, 164.4167, 168.475, 182.0417, 200.95, 207.2333, 134.725, 73.8, 53.5667, 39.8583], [6.1417, 27.0167, 46.8417, 61.7417, 64.2, 62.325, 60.8667, 57.3, 49.2333, 43.6333, 43.65, 41.6917, 39.975, 40.9667, 44.45, 50.1583, 57.1167, 59.0417, 59.1333, 58.4083, 54.6583, 49.3667, 42.4667, 37.0833, 36.7917, 38.7167, 42.8333, 47.3, 55.5417, 60.8833, 65.1417, 66.0, 59.725, 53.775, 47.1667, 36.3833, 24.8833, 36.5583, 78.9917, 100.3833, 121.6333, 142.4167, 163.3167, 188.55, 214.2083, 221.7917, 207.1333, 194.8333, 122.7917], [2.45, 7.9, 14.525, 22.7583, 29.625, 36.1, 41.0083, 44.025, 48.025, 51.625, 53.5583, 56.075, 56.6917, 61.8667, 68.4083, 77.025, 83.8417, 89.4083, 83.375, 73.4667, 66.3167, 58.8583, 47.4417, 39.2667, 34.5333, 29.45, 27.6667, 26.35, 25.2, 24.1083, 23.15, 22.0333, 19.975, 17.1167, 12.95, 9.4167, 5.7333, 80.7083, 161.6917, 134.225, 75.5333, 50.7333, 46.3, 51.875, 87.7583, 146.45, 214.6917, 265.5333, 175.825], [5.7667, 20.0167, 34.5, 44.3667, 56.5417, 61.025, 60.8417, 56.4833, 51.75, 45.075, 40.2833, 35.9833, 36.8, 40.3917, 46.4, 52.2583, 55.625, 60.05, 62.4167, 59.7, 53.8917, 50.1333, 44.6917, 38.5833, 35.3417, 36.9667, 42.3083, 47.8417, 54.45, 59.175, 56.7083, 50.8917, 41.1, 33.5, 28.7167, 22.1667, 17.0833, 56.5583, 132.325, 178.475, 184.4083, 156.5667, 136.5, 140.3667, 148.075, 152.4583, 151.5583, 124.35, 78.1833], [2.8333, 8.8083, 15.425, 22.2917, 29.7667, 35.7, 37.3917, 37.8, 36.7083, 36.0, 34.6083, 33.2333, 32.875, 33.5833, 34.15, 38.0, 42.3917, 50.4667, 62.1333, 73.7417, 83.625, 86.2583, 88.5917, 87.2, 84.3667, 79.825, 77.025, 74.0583, 70.75, 70.85, 72.8917, 73.65, 68.2417, 61.4167, 49.0583, 34.8083, 20.575, 103.5333, 259.7833, 255.2, 190.2333, 162.0917, 144.9167, 129.3833, 125.025, 122.9833, 131.2667, 151.7917, 104.8917], [4.8, 26.825, 52.7083, 73.3, 88.6, 96.5333, 86.55, 74.7917, 55.775, 41.1083, 32.625, 30.6083, 32.4333, 33.5917, 36.375, 41.2, 49.5333, 57.3583, 69.6333, 79.2667, 80.3833, 68.375, 57.6667, 47.3083, 39.175, 34.5667, 31.575, 28.675, 27.4167, 26.2083, 24.6667, 22.9333, 21.2167, 18.0833, 13.8083, 10.2, 7.1333, 30.2167, 58.9583, 78.0167, 105.55, 135.8667, 158.525, 186.15, 213.9917, 201.3833, 188.2083, 166.675, 99.4667], [6.65, 27.475, 48.825, 64.2583, 69.7667, 70.8083, 71.0333, 69.875, 66.225, 63.55, 64.6667, 67.175, 70.8, 76.9583, 81.5167, 82.0, 81.95, 80.45, 79.55, 79.4333, 77.1583, 72.5583, 66.8667, 62.7167, 60.825, 60.9917, 62.1333, 62.2833, 61.3417, 64.825, 64.5833, 65.5667, 63.875, 55.0417, 46.65, 34.8417, 20.1917, 71.0167, 189.15, 239.5583, 241.0917, 210.1583, 180.0083, 175.6917, 193.2417, 221.8917, 243.6667, 234.1, 125.8417], [7.375, 23.4583, 37.0417, 52.2833, 63.3667, 72.9167, 78.2167, 77.8167, 80.1083, 75.825, 74.2667, 76.0083, 76.15, 79.2083, 82.825, 
#82.9417, 86.25, 85.6583, 80.0333, 75.7167, 67.3417, 58.2833, 49.5, 40.875, 36.2167, 35.15, 34.175, 32.8083, 32.8167, 32.075, 30.9, 29.2417, 26.0083, 21.425, 16.2833, 11.3167, 6.85, 64.5583, 159.5083, 163.5833, 130.5583, 117.55, 115.1167, 116.0083, 127.475, 164.4167, 264.425, 335.875, 169.6583]]
#number = cv.imread("a.jpg", 0)
#print("El número es:", evalNumber(number, asd))
#plotImage(number)








#     """
#     number = cv.imread("0.jpg", 0)
#     hists = getHist4x4(number)

#     print(hists)

#     plt.figure(figsize=(8, 5))
#     plt.subplot(1,1,1)
#     plt.subplot(131)
#     plt.bar([1, 2, 3, 4], np.array(hists[0]))
#     plt.subplot(132)
#     plt.bar([1, 2, 3, 4], np.array(hists[1]))
#     plt.title('Original')
#     plt.show()

#     a = []
#     for i in range(0, 10):
#         a.append(getHist4x4(cv.imread(str(i) + ".jpg", 0)))
#     print(a)
#     print()

#     #histogramas para el dataset original resized to 150x50
#     A = [([533, 415, 428, 402], [516, 307, 270, 609]), ([518, 555, 337, 193], [111, 251, 790, 433]), ([93, 176, 224, 568], [201, 389, 351, 112]), ([300, 228, 413, 496], [136, 411, 509, 380]), ([385, 977, 383, 161], [522, 180, 381, 793]), ([572, 398, 212, 470], [246, 496, 435, 462]), ([299, 384, 879, 949], [806, 561, 537, 570]), ([784, 348, 413, 341], [137, 481, 783, 469]), ([512, 724, 440, 610], [369, 524, 516, 839]), ([588, 752, 409, 297], [260, 388, 566, 791])]

#     #histogramas para el dataset original5 resized to 150x50
#     B = [([344, 169, 210, 271], [332, 168, 174, 244]), ([690, 784, 706, 609], [151, 1459, 965, 210]), ([146, 187, 182, 429], [109, 184, 261, 359]), ([201, 226, 177, 403], [119, 227, 347, 258]), ([162, 280, 207, 106], [210, 60, 61, 410]), ([92, 98, 237, 248], [66, 275, 160, 177]), ([69, 180, 462, 520], [369, 340, 214, 286]), ([225, 179, 479, 114], [80, 218, 589, 96]), ([505, 506, 330, 246], [342, 385, 363, 465]), ([326, 420, 161, 111], [143, 105, 124, 589])]

#     #histogramas para el dataset original6 resized to 150x50
#     C = [([776, 553, 466, 688], [732, 493, 500, 687]), ([740, 1480, 1410, 330], [160, 1072, 1296, 1232]), ([96, 405, 362, 879], [331, 627, 623, 175]), ([0, 685, 560, 582], [212, 464, 631, 558]), ([197, 661, 847, 261], [538, 212, 195, 1025]), ([40, 821, 639, 589], [356, 569, 548, 578]), ([479, 533, 1018, 650], [774, 704, 551, 593]), ([714, 513, 613, 108], [154, 359, 452, 945]), ([723, 831, 753, 862], [496, 995, 883, 823]), ([684, 700, 736, 280], [544, 394, 410, 1010])]

#     #promedio de los 3 histogramas
#     [([551, 379, 368, 454], [527, 323, 315, 513]), ([649, 940, 818, 377], [141, 927, 1017, 625]), ([112, 256, 256, 625], [214, 400, 412, 215]), ([167, 380, 383, 494], [156, 367, 496, 399]), ([248, 639, 479, 176], [423, 151, 212, 743]), ([235, 439, 363, 436], [223, 447, 381, 406]), ([282, 366, 786, 706], [650, 535, 434, 483]), ([574, 347, 502, 188], [124, 353, 608, 503]), ([580, 687, 508, 573], [402, 635, 587, 709]), ([533, 624, 435, 229], [316, 296, 367, 797])]

#     D = []
#     #saca el promedio de los arreglos
#     for i in range(10):
#         innerH = []
#         innerV = []
#         for j in range(4):
#             innerH.append(round((A[i][0][j] + B[i][0][j] + C[i][0][j]) / 3))
#             innerV.append(round((A[i][1][j] + B[i][1][j] + C[i][1][j]) / 3))
#         D.append((innerH, innerV))
        
#     print(D)
# '''