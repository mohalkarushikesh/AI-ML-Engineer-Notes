import cv2
import numpy as np

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x, y), 100, (0, 0, 255), thickness=10)

img = cv2.imread('puppy.jpg')

cv2.namedWindow('dog')
cv2.setMouseCallback('dog', draw_circle)

while True:
    cv2.imshow('dog', img)
   
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
