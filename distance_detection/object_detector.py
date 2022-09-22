import cv2
from utils import *
import imutils

class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                if not is_contour_bad(cnt):
                    #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                    objects_contours.append(cnt)

        frame_ = frame.copy()
        cv2.drawContours(frame_, contours, -1, (0, 255, 0), 3)
        cv2.imshow("contours", frame_)
        return objects_contours

    def detect_objects_v2(self, frame):
        # convert to HSV, since red and yellow are the lowest hue colors and come before green
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # create a binary thresholded image on hue between red and yellow
        lower = (0,240,160)
        upper = (30,255,255)
        thresh = cv2.inRange(hsv, lower, upper)

        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # get external contours
        contours = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        cv2.imshow("thresh", thresh)
        cv2.imshow("clean", clean)
        return contours
    
    def detect_object_test(self, frame):
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       blurred = cv2.GaussianBlur(gray, (3, 3), 0)
       edged = cv2.Canny(blurred, 10, 100)

       # define a (3, 3) structuring element
       kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
       # apply the dilation operation to the edged image
       dilate = cv2.dilate(edged, kernel, iterations=1)

       # find the contours in the dilated image
       contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       frame_copy = frame.copy()
       # draw the contours on a copy of the original image
       cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 2)
       print(len(contours), "objects were found in this image.")

       cv2.imshow("Dilated image", dilate)
       cv2.imshow("contours", frame_copy)
    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)