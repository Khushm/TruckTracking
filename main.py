from utils.tracker import Tracker
import cv2
import os

execution_path = os.getcwd()

current_img = {'frame': cv2.imread(os.path.join(execution_path, "images", "1.jpg")), 'fileName': "20.jpg"}
previous_img = {'frame': cv2.imread(os.path.join(execution_path, "images", "11.jpg")), 'fileName': "21.jpg"}
Tracker(current_img, previous_img)
