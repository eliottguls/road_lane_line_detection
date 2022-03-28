import cv2
import numpy as np



image = cv2.imread('data/test_image.jpg') # Returns a multy dimensionnal array(relative iontensity of each pixel)
lane_image = np.copy(image) # all changements made in lane_image wiil be refelected in orginal image

# Edge detection = Identyfying sharp changes in intensity in adjacent pixels(O-255)
# Gradient = Measure of change in brightness over adjacent pixels
# Grayscale conversion is used here to pass pixel properties from 3 channels to 1(faster and simpler)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
cv2.imshow('Result', gray)
cv2.waitKey(0)










