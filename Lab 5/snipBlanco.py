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
    width = img.shape[1]
    

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
    width = img.shape[1]
    

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

img = cv.imread("a.jpg", 0)

pivoteColor = 200

dataset = []

crops = snipHorizontal(img)

filas = []
for crop in crops:
    filas.append(cropFila(img, crop))

for fila in filas:
    crops = snipVertical(fila)
    
    for crop in crops:
        dataset.append(cropColumn(fila, crop))

i = 1
for image in dataset:
    cv.imwrite(str(i) + ".jpg", image)
    i += 1





