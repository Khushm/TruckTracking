from imageai.Detection import ObjectDetection
import cv2
import os



execution_path = os.path.abspath("D:/InternShips/SmartIam/YOLOv3/")
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "model/yolo.h5"))
detector.loadModel()
all_files = os.listdir(os.path.join(execution_path, "images"))


def get_image(frame):
    truck_detections = []
    # if each_files.endswith(".jpg") or each_files.endswith(".jpeg"):
    # detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "images", each_files),
    #                                              output_image_path=os.path.join(execution_path, "output",
    #                                                                             each_files),
    #                                              minimum_percentage_probability=30)

    detections = detector.detectObjectsFromImage(input_image= frame,input_type="array",
                                                 output_image_path=os.path.join(execution_path, 'images' 'out_put.jpg'),
                                                 minimum_percentage_probability=30)
    #print(detections)
    return detections


def get_only_trucks(detections):
    Truck_box_plots=dict()
    truck_detections = []
    number_of_trucks = 0
    for obj in detections:
        boxplots = []
        if obj['name'] == 'truck':
            truck_detections.append(obj)
            #print(truck_detections[number_of_trucks]["box_points"])
            boxplots = truck_detections[number_of_trucks]["box_points"]
            try:
                Truck_box_plots['Co_Ordinates'].append(boxplots)
            except KeyError:
                Truck_box_plots['Co_Ordinates'] = [boxplots]
            number_of_trucks += 1
    Truck_box_plots['Count'] = number_of_trucks
    return Truck_box_plots

#def number_trucks(truck_detections):
 #   for eachObject in truck_detections:


def get_truck_detection(frames):
    detected_obj = get_image(frames)
    detected_truck = get_only_trucks(detected_obj)
    #print(detected_truck)
    return detected_truck

image_path = 'D:/InternShips/SmartIam/YOLOv3/images/1.jpg'
frame = cv2.imread(image_path)
get_truck_detection(frame)
#truck_count = number_trucks(detected_truck)

