from matplotlib import pyplot as plt
import os
from os import listdir, getcwd
import numpy as np
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

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
    i = 30
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

#recuperado de
#https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934
def crop_image(img, tol = 80):
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

def getHistsPromAndVariance(hists):

    prom = [0] * len(hists[0])

    for hist in hists:
        prom = np.add(prom, hist)

    prom = np.true_divide(prom, len(hists))

    variance = [0] * len(hists[0])
    for hist in hists:
        variance += np.abs(np.subtract(hist, prom))

    variance = np.true_divide(variance, len(hists))

    return np.around(prom, decimals = 4).tolist(), np.around(variance, decimals = 4).tolist()

def evalNumber(number, prom, variance):
    #pre procesamiento
    if(type(number) != list):
        ret, number = cv.threshold(number, 127, 255, cv.THRESH_BINARY)#se convierte a una imagen binaria 0 | 255
        number = crop_image(number)
        number = cv.resize(number, resizeDim, interpolation = cv.INTER_AREA)

        histNumber = getHist4x4(number, pixelWindowForHist)
    else:
        histNumber = number

    distances = []

    for hist in prom:
        dist = np.linalg.norm(np.array(histNumber) - np.array(hist))
        distances.append(dist)

    i = 0
    while(i < 10):
        predictedNumber = distances.index(min(distances))
        threshold = np.linalg.norm(np.array(variance[predictedNumber]) - np.array(prom[predictedNumber]))

        if(distances[predictedNumber] <= threshold):
            break
        else:
            distances[predictedNumber] = float('inf')

        i += 1

    return predictedNumber, histNumber

def readNumbers(readNumber, readFolder, pixelWindow):
    cwd = os.getcwd()
    pathTrain = cwd + "/" + readFolder
    trainFiles = os.listdir(path=pathTrain)

    hists = []
    y = []

    for imageName in trainFiles:
        number = cv.imread(readFolder + imageName, 0)
        number = cv.resize(number, resizeDim, interpolation = cv.INTER_AREA)
        hists.append((getHist4x4(number, pixelWindow)))
        y.append(readNumber)

    X_train, X_test, y_train, y_test = train_test_split(hists, y, test_size = 0.3, random_state = 42)

    return X_train, X_test, y_train, y_test

def generate_sub_dataset(readFolder, saveFolder, numberToRead):
    readFolder = readFolder + numberToRead 
    saveFolder = saveFolder + numberToRead + "/"
    readAndCropImage(cv.imread(readFolder + "/" + numberToRead + ".jpg", 0), saveFolder)

def getAccuracy(xTest, yTest, prom, var):

    total = len(xTest)
    successCount = 0
    for i in range(total):
        predictedNumber, _ = evalNumber(xTest[i], prom, var)
        if(yTest[i] == predictedNumber):
            successCount += 1

    return successCount / total

def plotResult(inputImg, predictedNumber, numHist, promHists):

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 6, 1)    
    plt.imshow(inputImg, cmap = 'gray')
    plt.title("Entrada, predicción: " + str(predictedNumber))

    plt.subplot(2, 6, 2)    
    plt.bar([i for i in range(len(numHist))], np.array(numHist))
    plt.title("Histograma de la entrada")

    for i in range(len(promHists)):
        plt.subplot(2, 6, i + 3)
        plt.bar([i for i in range(len(promHists[i]))], np.array(promHists[i]))
        plt.title("Promedio número: " + str(i))

    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()

def train_model():

    #saca los promedios de los números
    promHists = []
    varHists = []
    xTest = []
    yTest = []

    for i in range(0, 10):
        X_train, X_test, y_train, y_test = readNumbers(i, str(i) + "/", pixelWindowForHist)
        xTest = xTest + X_test
        yTest = yTest + y_test

        promI, varI = getHistsPromAndVariance(X_train)
        promHists.append(promI)
        varHists.append(varI)

    return promHists, varHists, xTest, yTest
#=============================================================================================================================
#Variables de control
pivoteColor = 200
resizeDim = (50, 150)#imagenes de 100 x 100 para los números individuales

pixelWindowForHist = 10
resizeDim = (int(resizeDim[0] / pixelWindowForHist) * pixelWindowForHist,
            int(resizeDim[1] / pixelWindowForHist) * pixelWindowForHist)
#=============================================================================================================================

#Entrenamiento
prom, var, xTest, yTest = train_model()
#=============================================================================================================================

#Accuracy
print("Accuracy:", getAccuracy(xTest, yTest, prom, var), "con una ventana de pixeles de:", pixelWindowForHist)
#=============================================================================================================================

#Predicciones
for i in range(10):
    num = cv.imread(str(i) + ".jpg", 0)
    predictedNumber, hist = evalNumber(num, prom, var)
    print("El numero es:", predictedNumber)
    plotResult(num, predictedNumber, hist, prom)
#=============================================================================================================================




"""
prom = [[151.0357, 228.8929, 212.5119, 167.7976, 136.631, 121.8452, 113.6786, 107.881, 108.2976, 114.7738, 125.4405, 149.1429, 187.9524, 234.1667, 179.9524, 622.131, 408.7024, 370.7381, 400.3214, 538.1071], [233.5, 301.4881, 315.7262, 325.4643, 322.369, 324.0714, 326.0833, 322.8214, 316.0476, 315.9405, 314.5476, 312.7262, 304.5595, 290.9881, 231.3333, 405.2857, 986.8333, 1300.2143, 1228.1071, 637.2262], [133.75, 161.1071, 134.1786, 98.1429, 67.9643, 56.0238, 56.2738, 58.5, 68.869, 108.381, 124.0119, 129.9762, 199.1429, 280.619, 189.1548, 330.6548, 460.25, 549.0833, 393.7024, 132.4048], [193.881, 211.5833, 129.2262, 83.5595, 88.7381, 114.0476, 153.4762, 178.2262, 123.3452, 78.7738, 76.3452, 103.131, 146.9405, 
233.6071, 227.131, 205.6071, 355.0476, 518.3214, 585.7024, 477.3333], [68.9643, 114.6905, 131.5714, 137.0238, 138.4643, 170.6548, 248.131, 255.119, 136.869, 84.6786, 66.3929, 64.1429, 63.369, 63.6429, 53.9762, 398.9643, 196.25, 148.5476, 407.5238, 646.4048], [195.2976, 231.619, 129.2857, 67.2143, 68.2976, 101.8333, 168.119, 193.4762, 140.3214, 79.0357, 66.7738, 88.8571, 141.8571, 208.0833, 187.3214, 357.3452, 518.631, 417.4167, 451.2857, 322.7143], [92.4524, 120.9286, 103.9405, 94.9405, 90.2857, 89.0357, 98.3452, 180.6548, 268.369, 259.1548, 212.5952, 193.6905, 215.119, 278.131, 216.369, 709.881, 562.881, 438.0119, 386.7262, 416.5119], [249.8333, 301.2024, 131.5476, 75.4405, 72.4048, 79.0952, 105.6429, 202.1905, 246.4405, 130.3452, 80.9524, 72.2619, 72.631, 70.7381, 57.3333, 159.7619, 316.1548, 511.869, 553.9405, 406.3333], [180.0, 219.7381, 171.7381, 150.0714, 156.7262, 197.5952, 238.7619, 234.4405, 192.881, 150.0476, 135.6071, 140.8095, 164.7976, 222.5357, 214.0833, 484.869, 637.4048, 504.3214, 603.381, 539.8571], [186.8095, 241.7738, 194.381, 172.3571, 180.2738, 213.1667, 253.619, 229.4405, 154.2976, 103.75, 85.5238, 83.381, 85.881, 90.4643, 73.0, 407.0, 354.5714, 329.0238, 490.9405, 766.5833]]

variance = [[25.227, 45.8605, 59.3475, 45.5017, 31.8878, 25.5723, 22.5842, 21.9229, 20.9167, 23.4297, 29.4005, 43.483, 65.2358, 53.1111, 31.1916, 90.0193, 129.3452, 98.47, 81.2976, 77.6046], [50.4167, 60.6182, 65.6786, 65.2755, 68.6584, 69.0017, 64.3889, 62.477, 63.3821, 62.5876, 60.3333, 57.2846, 54.049, 49.8467, 46.0952, 206.9388, 285.5317, 205.8265, 172.1871, 224.0964], [24.5714, 64.1709, 62.5944, 46.5782, 26.1241, 15.6905, 13.9014, 15.7857, 22.9436, 58.1054, 63.5601, 59.6701, 
83.4252, 99.1576, 44.4351, 105.8954, 149.7857, 171.6171, 171.1122, 42.0142], [28.1429, 108.254, 76.7747, 28.809, 22.1655, 27.1644, 46.0578, 86.6678, 63.801, 23.2914, 20.4668, 38.4354, 61.9082, 75.5408, 37.3194, 67.4651, 137.1338, 145.4694, 198.1502, 119.0317], [25.6752, 46.1429, 54.881, 59.3107, 57.642, 69.1392, 119.4592, 128.3571, 84.2075, 41.6097, 25.9311, 23.9762, 23.7262, 23.7398, 18.0743, 136.0408, 118.1131, 69.7795, 284.1553, 198.3764], [68.1497, 120.6757, 89.4422, 28.3248, 23.557, 44.9286, 81.1837, 105.822, 82.5816, 30.2806, 23.4697, 41.7755, 76.2109, 97.4583, 33.6939, 221.1638, 192.7112, 186.5079, 176.9048, 114.5884], [35.5187, 58.2109, 30.3087, 19.5315, 14.466, 13.3963, 24.3265, 75.7336, 73.5833, 61.5394, 49.7188, 39.7103, 50.3475, 58.6817, 30.2024, 105.9666, 154.3628, 182.0592, 127.3214, 60.2622], [80.5, 115.5451, 80.9484, 22.07, 20.1905, 26.3878, 49.6497, 73.9342, 85.049, 74.627, 27.61, 18.7619, 18.4016, 17.3022, 13.6667, 52.9127, 145.1074, 148.6071, 172.3705, 346.1667], [25.0238, 41.5476, 36.5578, 29.4303, 30.2132, 45.2942, 50.7732, 59.8271, 52.2506, 29.7098, 22.6105, 23.1463, 40.2594, 63.8231, 26.7262, 145.1267, 137.0289, 95.4337, 94.9286, 117.1905], [23.4751, 70.2846, 46.9615, 32.2789, 42.8387, 62.2183, 66.2528, 87.9272, 80.79, 40.6607, 24.6088, 22.6429, 24.322, 27.0731, 15.2381, 74.9762, 144.8946, 143.4575, 179.6953, 165.3294]]
"""