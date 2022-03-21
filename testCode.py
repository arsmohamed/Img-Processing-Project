import cv2
import imutils
import numpy as np
from skimage.util import random_noise
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread(r"F:\projects\Im-Proc-Proj\Project\testimg2.jpeg",cv2.IMREAD_COLOR)
width = 600 # keep original width
height = 440
dim = (width, height)
#resizing the image
img_resized = cv2.resize(img, dim) 
cv2.imshow('Original Image',img_resized)

# adding gray scale to avoid dealing with color details
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('Gray scale',gray)

# Removing the nosie from the image
remove_noise = cv2.GaussianBlur(gray,(7,7),0)
cv2.imshow('Removing Noise',remove_noise)

# smoothing the image from noise, edge preserving 
gray = cv2.bilateralFilter(gray, 13, 15, 15) 
cv2.imshow('Bilateral Filter',gray)

# Threshold detection using 0 and 150 output array 
ret2,out_binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Thresholding image (Binary)', out_binary)

ret, out_binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholding image (Binary) 2', out_binary)

# edge detection 
edged = cv2.Canny(out_binary, 30, 200) 
cv2.imshow('Edge detected 1',edged)


# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
#  # 
# dilate=cv2.dilate(out_binary,kernel)
#  # 
# erode=cv2.erode(out_binary,kernel)
#  # 
# edged=cv2.absdiff(dilate,erode)
# cv2.imshow('Edge detected 1',edged)

contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in contours:
    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
    if len(approx) == 4:
        screenCnt = approx
        break

# checking if there is no contour 
if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

# The mask propirty 
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

cv2.imshow('Masked img',new_image)

# croped image
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]
Cropped = cv2.resize(Cropped,(400,200))
cv2.imshow('Cropped',Cropped)

# character recognition
text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("programming_fever's License Plate Recognition\n")
print("Detected license plate Number is:",text)

cv2.waitKey(0)
cv2.destroyAllWindows()