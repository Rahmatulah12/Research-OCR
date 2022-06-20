from array import array
from easyocr import Reader
import pytesseract
import cv2
import numpy as np

# load the image and resize it
image = cv2.imread('./train_data2/truk_new3.jpg')
image = cv2.resize(image, (670, 500))
# n_plate_cnt = np.arange(2).reshape(2, 2)
n_plate_cnt = np.array([[430, 430], [430, 430]])

# convert the input image to grayscale,
# blur it, and detect the edges 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
blur = cv2.GaussianBlur(gray, (5,5), 0) 
edged = cv2.Canny(blur, 10, 200) 

# find the contours, sort them, and keep only the 5 largest ones
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in contours:
    # approximate each contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if the contour has 4 points, we can say
    # that we have found our license plate
    if len(approx) == 4:
        n_plate_cnt = approx
        break        

# # get the bounding box of the contour and 
# # extract the license plate from the image
(x, y, w, h) = cv2.boundingRect(n_plate_cnt)
license_plate = gray[y:y + h, x:x + w]

cv2.imshow('Image', license_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # initialize the reader object
# reader = Reader(['en'], gpu=False)
# # detect the text from the license plate
# detection = reader.readtext(license_plate)
detection = pytesseract.image_to_string(license_plate,config='--psm 7, --oem 3', lang="eng")

if len(detection) == 0:
    # if the text couldn't be read, show a custom message
    text = "Impossible to read the text from the license plate"
    cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
else:
    # draw the contour and write the detected text on the image
    cv2.drawContours(image, [n_plate_cnt], -1, (0, 255, 0), 3)
    # text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
    text=detection
    print(text)
    cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # display the license plate and the output image
    cv2.imshow('license plate', license_plate)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
