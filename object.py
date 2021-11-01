from imageai.Detection import ObjectDetection
from PIL import Image
import cv2
import os
import numpy as np

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "model/yolo.h5"))
detector.loadModel()

all_files = os.listdir(os.path.join(execution_path, "images"))

result = {
    1: {'Area': 10117291, 'Centroid': [50, 50], 'Coordinates': (0, 0, 0, 0), 'Image_name': 'img1.jpg'},
}

maxTruckSize = 65000
Med_THV = 15
Area_THV = 102000
i=1


def Centroid(med, area, x1,y1,x2,y2, each_files):
    find = False
    count = 1
    for key, value in result.items():
        if (value['Centroid'][0] + Med_THV > med[0] > value['Centroid'][0] - Med_THV) and (
                value['Centroid'][1] + Med_THV > med[1] > value['Centroid'][1] - Med_THV) and (
                value['Area'] + Area_THV > area > value['Area'] - Area_THV):
            result[key]['Image_name'].append(each_files)
            # value['Time'] += 15
            print('ID: ', key)
            find = True
        count += 1
    if (find == False):
        dict1 = {count: {'Area': area, 'Centroid': med, 'Coordinates': (x1, y1, x2, y2), 'Image_name': [each_files]}}
        print('ID: ', count)
        result.update(dict1)


for each_files in all_files:
    truck_detections = []
    # if each_files.endswith(".jpg") or each_files.endswith(".jpeg"):
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "images", each_files),
                                                 output_image_path=os.path.join(execution_path, "output",
                                                                                each_files),
                                                 minimum_percentage_probability=30)
    frame = cv2.imread(os.path.join(execution_path, "images", each_files), 0)
    if frame is None:
        print("Image None: ", each_files, all_files)
        continue
    else:
        print("Frame is true: ", each_files, all_files)

    for obj in detections:
        if obj['name']== 'truck':
            truck_detections.append(obj)
    obj_counter = 0
    for eachObject in truck_detections:
        # print(eachObject)
        x1, y1, x2, y2 = eachObject["box_points"]
        current = frame
        # current = frame[int(y1):int(y2), int(x1):int(x2)]
        cv2.imshow("Current", current)

        med = ((x1 + x2) / 2, (y1 + y2) / 2)
        area = ((x2 - x1) * (y2 - y1))

        # if eachObject["name"] == 'truck':
        # print("Type:::", type(eachObject))
        if (area > maxTruckSize):
            Centroid(med, area, x1, y1, x2, y2, each_files)

            if(i==1):
                i+=1
                previous = current
                continue

            # if(len(truck_detections) > 1):
            #     obj_counter += 1
            #     if(obj_counter == len(truck_detections)):
            #         print(obj_counter, len(truck_detections),'))))))))))))))))))')
            #         previous = current
            # else:
            #     previous = current


            # obj_counter += 1
            # if(obj_counter == len(truck_detections)):
            #     print(obj_counter, len(truck_detections),'))))))))))))))))))')
            #     previous = current

            ch, cw = current.shape
            ph, pw = previous.shape

            # print(ch,cw)
            # print(ph,pw)

            # resizing both images
            if(ch*cw > ph*pw):
                print("resize")
                current = cv2.resize(current, (pw, ph))
            else:
                print("resize2")
                previous = cv2.resize(previous, (cw, ch))

            print(current.shape, previous.shape)


            # difference = cv2.subtract(current, previous)
            #
            # Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            # ret, mask = cv2.threshold(Conv_hsv_Gray, 62, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            # difference[mask != 255] = [0, 0, 255]

            # add the red mask to the images to make the differences obvious
            # previous[mask != 255] = [0, 0, 255]
            # current[mask != 255] = [0, 0, 255]

            diff = cv2.absdiff(current, previous)
            _, diff = cv2.threshold(diff, 32, 0, cv2.THRESH_TOZERO)
            _, diff = cv2.threshold(diff, 62, 255, cv2.THRESH_BINARY)

            diff = cv2.medianBlur(diff, 5)
            if not os.path.exists(str(each_files.split('.')[0])):
                os.mkdir(str(each_files.split('.')[0]))
            cv2.imwrite(str(each_files.split('.')[0])+'/current'+str(obj_counter)+str(each_files.split('.')[0])+'.jpg', current)
            cv2.imwrite(str(each_files.split('.')[0])+'/prev'+str(obj_counter)+str(each_files.split('.')[0])+'.jpg', previous)
            cv2.imwrite(str(each_files.split('.')[0])+'/Difference'+str(obj_counter)+str(each_files.split('.')[0])+'.jpg', diff)

            white_count = np.sum(diff == 255)
            black_count = np.sum(diff == 0)
            print("white, black", white_count, black_count)


            print(each_files)
            print(result)
            # print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
            print("Area: ", area)
            # print("Median: ", med)
            # obj_counter += 1
            print(obj_counter, len(truck_detections)-1, '))))))))))))))))))')
            if (obj_counter == len(truck_detections)-1):
                print('Swaping Frames')
                previous = current
            obj_counter += 1
    # print("Sizeee: ", each_files.shape)
    print("--------------------------------")
# white, black 172810 776357
# data = [
#     {
#         "id": 1,
#         "area": 4000,
#         "cordinates": [
#             [34, 66],
#             [34, 66],
#             [34, 66],
#             [34, 66]
#         ],
#         "centroid":[99,77],
#         "time":{
#             "10am":True,
#             "11am":False,
#             "12pm":True
#         }
#     }
# ]
