import cv2
import pyrealsense2
from realsense_depth import *
from time import sleep
from imutils import perspective
from datetime import datetime

def get_Datetime():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def draw_circle_and_find_distance(arg_point, depth_frame, color_frame, point_distance_in_cm):
    distance_point_1 = depth_frame[arg_point[1], arg_point[0]]
    distance_point_1_in_cm = distance_point_1/10
    diff_dist = distance_point_1_in_cm-point_distance_in_cm
    if diff_dist > 0:
        cv2.circle(color_frame, arg_point, 4, (0, 0, 255))
        cv2.putText(color_frame, "{}cm".format(distance_point_1_in_cm), (arg_point[0] - 50, arg_point[1] - 20), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2)
        cv2.putText(color_frame, "diff dist:{}cm".format(int(diff_dist)),(arg_point[0] - 50, arg_point[1] - 45), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2)
        return True, diff_dist
    else:
        return False, diff_dist

def is_contour_bad(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4

def draw_detected_obj_boundingbox(color_frame, detector):
    thickness = 2
    color = (0, 255, 0)
    contours = detector.detect_objects(color_frame)
    # detector.detect_object_test(color_frame)

    if len(contours) != 0:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect

        # Display rectangle
        box = cv2.boxPoints(rect)
        box = perspective.order_points(box)
        box = np.int0(box)
        cv2.drawContours(color_frame, [box.astype("int")], -1, color, thickness)

def round_num(x):
    i, f = divmod(x, 1)
    return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))

def apply_colormap(depth_image, color_image):
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        images = np.hstack((resized_color_image, depth_colormap))
    else:
        images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
