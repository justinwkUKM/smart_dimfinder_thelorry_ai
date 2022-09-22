import cv2
from object_detector import *
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import paths
import numpy as np
from time import sleep
from config import PIXELS_PER_METRIC

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def process_img(img=None, detector=None, cm_height=None, height_marker_position=(0,0)):
    # Get Aruco marker
    try:
        corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        # markerSizeInCM = 5
        # rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, mtx, dist)
        if corners:
            # print('marker found!')
            # Draw polygon around the marker
            int_corners = np.int0(corners)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

            # Aruco Perimeter
            aruco_perimeter = cv2.arcLength(corners[0], True)

            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / PIXELS_PER_METRIC

            contours = detector.detect_objects(img)

            # Draw objects boundaries
            for cnt in contours:
                # Get rect
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect

                # Get Width and length of the Objects by applying the Ratio pixel to cm
                object_width = w / pixel_cm_ratio
                object_length = h / pixel_cm_ratio

                # Display rectangle
                box = cv2.boxPoints(rect)
                box = perspective.order_points(box)
                box = np.int0(box)
                cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)

                # loop over the original points and draw them
                for (x, y) in box:
                    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box

                # print("(tl, tr, br, bl) :=> ",(tl, tr, br, bl))
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the top-left and bottom-left points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # draw the midpoints on the image
                cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                # draw lines between the midpoints
                # cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                #     (255, 69, 0), 2)
                # cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                #     (255, 69, 0), 2)

                object_length = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                object_width = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))


                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                # cv2.polylines(img, [box], True, (255, 0, 0), 2)
                # cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                # cv2.putText(img, "length {} cm".format(round(object_length, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                # compute the size of the object
                cm_length = object_length / pixel_cm_ratio
                cm_width = object_width / pixel_cm_ratio

                # draw the object sizes on the image
                cv2.arrowedLine(img, (tr[0], tr[1]), (tl[0], tl[1]),(255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(img, (tr[0], tr[1]), (br[0], br[1]),(255, 0, 255), 3, 8, 0, 0.05)


                cv2.putText(img, "width: {:.1f}cm".format(cm_width),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_ITALIC,
                    0.65, (0, 0, 255), 2)
                cv2.putText(img, "length: {:.1f}cm".format(cm_length),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_ITALIC,
                    0.65, (0, 0, 255), 2)

                # cv2.putText(img, "height: {}cm".format(cm_height), (height_marker_position[0]-40, height_marker_position[1]+45), cv2.FONT_ITALIC,
                #     0.65, (0, 0, 255), 2)
                # cv2.putText(img, "height: {:.1f}cm".format(cm_height),
                #     (int(blbrX + 10), int(blbrY)), cv2.FONT_ITALIC,
                #     0.65, (0, 0, 255), 2)
            
            # cv2.imshow(window_name, img)

    except Exception as e:
        print('error found =>', e)
        sleep(2)
        pass



# def find_marker(image):
# 	# convert the image to grayscale, blur it, and detect edges
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	gray = cv2.GaussianBlur(gray, (5, 5), 0)
# 	edged = cv2.Canny(gray, 35, 125)
# 	# find the contours in the edged image and keep the largest one;
# 	# we'll assume that this is our piece of paper in the image
# 	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 	cnts = imutils.grab_contours(cnts)
# 	c = max(cnts, key = cv2.contourArea)
# 	# compute the bounding box of the of the paper region and return it
# 	return cv2.minAreaRect(c)


# def distance_to_camera(knownWidth, focalLength, perWidth):
# 	# compute and return the distance from the maker to the camera
# 	return (knownWidth * focalLength) / perWidth