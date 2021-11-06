from utils.detector import get_truck_detection
import cv2
import numpy as np
import os

execution_path = os.getcwd()

truck_coord = {'Coordinates': [[0, 0, 0, 0], [0, 0, 0, 0]], 'TruckCount': 2}
main_response = []

def area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


def maxTruckSize():
    value = 65000
    return value


def cropImg(current_img, x1, y1, x2, y2):
    return current_img[y1:y2, x1:x2]


def backgroundSubtract(current_, previous_):

    current = cv2.cvtColor(current_, cv2.COLOR_BGR2GRAY)
    previous = cv2.cvtColor(previous_, cv2.COLOR_BGR2GRAY)

    # current = cv2.Canny(current_, 100, 200)
    # previous = cv2.Canny(previous_, 100, 200)

    diff = cv2.absdiff(current, previous)

    _, diff = cv2.threshold(diff, 32, 0, cv2.THRESH_TOZERO)
    _, diff = cv2.threshold(diff, 62, 255, cv2.THRESH_BINARY)
    diff = cv2.medianBlur(diff, 5)
    cv2.imwrite(outputDir + '/Diff' + str(i) + '.jpg', diff)

    # contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # detections = []
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area>700:
    #         cv2.drawContours(current_, [cnt], -1, (255, 0, 0), 2)
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         cv2.rectangle(current_, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #         detections.append([x, y, w, h])

    h, w = diff.shape
    tot_pix_count = h*w
    print('Total Count of Pixesl is', tot_pix_count)
    # Count = [np.sum(diff == 255), np.sum(diff == 0)]
    # {"black_pix_count": "", "white_pixel_count": ""}
    Count = {"black_pixel_count": float(np.sum(diff == 0)/tot_pix_count) * 100, "white_pixel_count": float(np.sum(diff == 255)/tot_pix_count) * 100}
    print(Count)
    return Count

def count(diff):
    return np.sum(diff)


def Tracker(current_img, previous_img):
    global outputDir
    outputDir = os.path.join(execution_path, 'output', current_img['fileName'].split('.')[0])
    print(outputDir)

    cv2.imwrite(str(outputDir) + '/Current.jpg', current_img['frame'])
    cv2.imwrite(outputDir + '/Previous.jpg', previous_img['frame'])

    response = {current_img['fileName']: []}

    truck_coord_ = get_truck_detection(previous_img['frame'])
    global i
    i = 0

    if len(truck_coord_["Co_Ordinates"]) <= 0:
        print("No detections Found!")
        return main_response

    print(truck_coord_)
    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords

        if area(x1, y1, x2, y2) > maxTruckSize():
            current_frame = cropImg(current_img['frame'], x1, y1, x2, y2)
            previous_frame = cropImg(previous_img['frame'], x1, y1, x2, y2)

            cv2.imwrite(outputDir + '/CropCurrent' + str(i) + '.jpg', current_frame)
            cv2.imwrite(outputDir + '/CropPrevious' + str(i) + '.jpg', previous_frame)

            # cv2.imwrite('current.jpg', current_frame)
            # cv2.imwrite('previous.jpg', previous_frame)

            Count = backgroundSubtract(current_frame, previous_frame)
            i += 1
            individual_objects = {i: Count}
            response[current_img['fileName']].append(individual_objects)
            # truck_coord['Count'] -= 1
    main_response.append(response)
    print(main_response)
    return main_response


# main_data = {"image_current": [{id: {"black_pix_count": "", "white_pixel_count": ""}}, {id: {...}}..... ]