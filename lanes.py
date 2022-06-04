import cv2
import numpy as np


def canny(image):
# Grayscale conversion is used here to pass pixel properties from 3 channels to 1(faster and simpler)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0] # canny array is 2 dimensional (m,n)

    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ]) # (bottom left corner, bottom right corner, top corner)

    mask = np.zeros_like(image) # fullfil an array with zeros(black pixel). The array created here has the same number of rows and columns than the array of our image
    cv2.fillPoly(mask, polygons, 255) # 255 is white
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

image = cv2.imread('data/test_image.jpg') # Returns a multy dimensionnal array(relative iontensity of each pixel)
lane_image = np.copy(image) # all changements made in lane_image wiil be refelected in orginal image
canny = canny(lane_image)
cropped_image = region_of_interest(canny)


# It's used to get a view of positions with axis x and y of the points we need
cv2.imshow('Result', cropped_image)
cv2.waitKey(0)

'''
OR
plt.imshow(canny)
plt.show()
It's used to get a view of positions with axis x and y of the points we need
'''









