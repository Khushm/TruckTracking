from imageai.Detection import ObjectDetection
import cv2
import os

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "model/yolo.h5"))
detector.loadModel()


def get_image(img1):


