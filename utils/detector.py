from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "model/yolo.h5"))
detector.loadModel()
all_files = os.listdir(os.path.join(execution_path, "images"))


def get_image(frame):
    detections = detector.detectObjectsFromImage(input_image=frame, input_type="array",
                                                 output_image_path=os.path.join(execution_path, 'images' 'out_put.jpg'),
                                                 minimum_percentage_probability=30)
    return detections


def get_only_trucks(detections):
    Truck_box_plots = dict()
    truck_detections = []
    number_of_trucks = 0
    for obj in detections:
        if obj['name'] == 'truck':
            truck_detections.append(obj)
            boxplots = truck_detections[number_of_trucks]["box_points"]
            try:
                Truck_box_plots['Co_Ordinates'].append(boxplots)
            except KeyError:
                Truck_box_plots['Co_Ordinates'] = [boxplots]
            number_of_trucks += 1
    Truck_box_plots['Count'] = number_of_trucks
    return Truck_box_plots


def get_truck_detection(frames):
    detected_obj = get_image(frames)
    detected_truck = get_only_trucks(detected_obj)
    return detected_truck
