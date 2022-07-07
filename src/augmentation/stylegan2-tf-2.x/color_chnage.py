import cv2
import numpy as np

# read image
img = cv2.imread('car.jpg')

# convert to SHV
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# threshold for Saturation
th_s=18
# threshold for Value

print(img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# threshold on gray color
# lower=(0,0,0)
# upper=(70,255,255)
# mask = cv2.inRange(img, lower, upper)

# # change all non-yellow to white
# result = img.copy()
# result[mask!=255] = (255, 255, 255)

# # save results
# cv2.imwrite('car.jpg',result)

# # display result
# cv2.imshow("mask", mask)
# cv2.imshow("result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




