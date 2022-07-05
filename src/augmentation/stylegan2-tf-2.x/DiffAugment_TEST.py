import cv2
import numpy as np

# read image
img = cv2.imread('car.jpg')

# threshold on gray color
lower=(0,0,0)
upper=(70,255,255)
mask = cv2.inRange(img, lower, upper)

# change all non-yellow to white
result = img.copy()
result[mask!=255] = (255, 255, 255)

# save results
cv2.imwrite('corn_yellow.jpg',result)

# display result
cv2.imshow("mask", mask)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()




