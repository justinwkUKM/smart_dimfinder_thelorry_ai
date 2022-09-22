import cv2
import pyrealsense2
from realsense_depth import *
from statistics import mean
from object_detector import HomogeneousBgDetector
from utils import *
from aruco_utils import *

point = (400, 300)

MAX_X_WINDOW = 640
MAX_Y_WINDOW = 480

center_marker = (int(MAX_X_WINDOW/2), int(MAX_Y_WINDOW/2))
center_marker = (int(MAX_X_WINDOW/2), int(MAX_Y_WINDOW/2))

mid_btm_marker = (int((MAX_X_WINDOW/9)*5), int((MAX_Y_WINDOW/6)*5))
mid_top_marker = (int((MAX_X_WINDOW/9)*5), int((MAX_Y_WINDOW/6)*1))
mid_l_marker = (int((MAX_X_WINDOW/9)*2), int((MAX_Y_WINDOW/6)*3))
mid_r_marker = (int((MAX_X_WINDOW/9)*8), int((MAX_Y_WINDOW/6)*3))
mid_rtop_marker = (int((MAX_X_WINDOW/9)*8), int((MAX_Y_WINDOW/6)*1))
mid_ltop_marker = (int((MAX_X_WINDOW/9)*2), int((MAX_Y_WINDOW/6)*1))
mid_lbtm_marker = (int((MAX_X_WINDOW/9)*2), int((MAX_Y_WINDOW/6)*5))
mid_rbtm_marker = (int((MAX_X_WINDOW/9)*8), int((MAX_Y_WINDOW/6)*5))

marker_list = [mid_rtop_marker, mid_top_marker, mid_ltop_marker, mid_l_marker, mid_lbtm_marker, mid_btm_marker, mid_rbtm_marker, mid_r_marker]

dc = DepthCamera()
detector = HomogeneousBgDetector()

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

while True:
    ret, depth_frame, color_frame = dc.get_frame()

    # Show distance for a specific point
    cv2.circle(color_frame, point, 4, (0, 0, 255))

    distance = depth_frame[point[1], point[0]]
    point_distance_in_cm = distance/10
    height_avg_diff_dist = 0
    diff_dist_list = []
    for marker in marker_list:
        res, dd = draw_circle_and_find_distance(marker, depth_frame, color_frame, point_distance_in_cm)
        if res:
            diff_dist_list.append(dd)
    print(diff_dist_list)

    if len(diff_dist_list) > 0:
        height_avg_diff_dist = round_num(mean(diff_dist_list))

    process_img(img=color_frame, detector=detector, cm_height=height_avg_diff_dist, height_marker_position=mid_top_marker)

    # cv2.putText(color_frame, "Place desired object inside the bounding box below.", (mid_top_marker[0]-220, mid_top_marker[1]-45), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2)
    cv2.putText(color_frame, "PRESS 'S' TO SAVE IMAGE.", (mid_btm_marker[0]-100, mid_btm_marker[1]+20), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2)

    # cv2.putText(color_frame, "{}cm".format(point_distance_in_cm), (point[0], point[1] - 20), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 2)
    cv2.putText(color_frame, "height: {}cm".format(height_avg_diff_dist), (point[0], point[1] - 45), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 2)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
    draw_detected_obj_boundingbox(color_frame, detector)
    cv2.rectangle(color_frame, mid_ltop_marker, mid_rbtm_marker, (255, 0, 0), 2)
    apply_colormap(depth_frame, color_frame)

    # cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Color frame", color_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif (key == ord('s')) or (key == ord('S')): # wait for 's' key to save and exit
        image_save_path = f'img/package_dimweight_{get_Datetime()}.png'
        cv2.imwrite(image_save_path,color_frame)
        # sleep(3)
        cv2.destroyAllWindows()
        print(f'Image Saved! => {image_save_path}')
        break