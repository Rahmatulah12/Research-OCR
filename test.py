import pytesseract
import matplotlib.pyplot as plt
import cv2
from easyocr import Reader

car_cascade = cv2.CascadeClassifier('cascade.xml')

image = cv2.imread('train_data2/ktp3.jpg')
image = cv2.resize(image, (500, 400), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.GaussianBlur(image,(5,5),-1)
image = cv2.bilateralFilter(image, 9, 75, 75)
th, threshed = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
reader = Reader(['en'], gpu=False)
text = reader.readtext(image)
# print(f"{text[0][1]} {text[0][2] * 100:.2f}%")
if(len(text) > 0):
    print(text)
    print(f"{text[0][1]}")
else:
    print("No text found")
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()