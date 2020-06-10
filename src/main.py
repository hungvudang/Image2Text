import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
WIDTH = 640
HEIGHT = 480

def displayImg(image, nameWindow="Image View"):
    cv.imshow(nameWindow, image)
    cv.waitKey(0)


img = cv.imread(r"image\test1.jpg",1)
img = cv.resize(img, (WIDTH, HEIGHT))

# Chuyển thành gray
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# cv.imshow("Image View", img)
# cv.waitKey(0);

blurImg = cv.GaussianBlur(grayImg, (9,9), 0, 0)
# displayImg(blurImg)

thImg = cv.adaptiveThreshold(blurImg, 250, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 3, 2)
# displayImg(thImg)
# Remove noises
kernel = np.ones((3,3), np.uint8)
openImg = cv.morphologyEx(thImg, cv.MORPH_OPEN, kernel)
kernel = np.ones((5,30), np.uint8)
closeImg = cv.morphologyEx(openImg, cv.MORPH_CLOSE, kernel)
# displayImg(closeImg)

# Find contours cho từng dòng văn bản trong ảnh
cnt, _ = cv.findContours(closeImg,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# print(cnt)

lines = [] # Lưu tọa đọ từng dòng văn bản đc tìm thấy

for c in cnt:
    x,y,w,h = cv.boundingRect(c)
    lines.extend([[x,y,w,h]])
    # cv.rectangle(img,(x,y), (x+w-1, y+h-1),(0,255,0),1)
# displayImg(img)



# Hiện thị từng dòng trong văn bản
kernel =np.array([[0,1,1,0,0],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [0,1,1,0,0]],np.uint8)

# kernel = np.ones((2,4), np.uint8)

for x,y,w,h in reversed(lines):
    # Find contours cho từng chữ trong ảnh
    line = thImg[y:y+h, x:x+w]
    line = cv.morphologyEx(line, cv.MORPH_CLOSE,kernel)
    cnt, _ = cv.findContours(line,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for c in cnt:
        if cv.contourArea(c) < 15:
            continue
        
        x1,y1,w1,h1 = cv.boundingRect(c)

        # Vẽ bbox cho từng chữ trong ảnh
        cv.rectangle(img[y:y+h, x:x+w],(x1,y1), (x1+w1-1, y1+h1-1),(0,0,255),1)
    # displayImg(line,"Line View ")
displayImg(img)
# git config --global user.email "you@example.com"
# git config --global user.name "Your Name"