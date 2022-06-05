import cv2
import numpy as np


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit =  [] # contains slope's value on the right of our screen
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        ''' 
        y = mx + b
        m = (y2-y1)/(x2-x1)
        b = y - mx
        '''
        parameters = np.polyfit((x1,x2), (y1,y2), 1) # will return a vector of coefficient which describe the slope and y intercept values (linear fonction)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
# Grayscale conversion is used here to pass pixel properties from 3 channels to 1(faster and simpler)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50,150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0] # canny array is 2 dimensional (m,n)

    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ]) # (bottom left corner, bottom right corner, top corner)

    mask = np.zeros_like(image) # fullfil an array with zeros(black pixel). The array created here has the same number of rows and columns than the array of our image
    cv2.fillPoly(mask, polygons, 255) # 255 is white
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image


''' PROCESS FOR 1 IMAGE :
image = cv2.imread('data/test_image.jpg') # Returns a multy dimensionnal array(relative iontensity of each pixel)
lane_image = np.copy(image) # all changements made in lane_image wiil be refelected in orginal image
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 
# 4th arguments = threshold  : minimum number of votes needed to accept a candidate line
# minLineLength : Length of lines minimum required to note reject them
# maxLineGap : maximum distance (pixels) between segemented lines which will be able to be connected into a single line instead of multiple lines intead of them being broken up apart
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # multiply the arrays to get opacity reduced on the lines we want ot draw


# It's used to get a view of positions with axis x and y of the points we need
cv2.imshow('Result', combo_image)
cv2.waitKey(0)

OR
plt.imshow(canny)
plt.show()
It's used to get a view of positions with axis x and y of the points we need
'''

cap = cv2.VideoCapture("data/video.mp4")
while(cap.isOpened):
    _, frame = cap.read() # Decode every video frame
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('Result', combo_image)
    if cv2.waitKey(1) == ord('q'):# wait 1 ms before printing next frame 
        break

cap.release()
cv2.destroyAllWindows()





