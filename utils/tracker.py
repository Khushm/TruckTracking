import time
from globals import config

from utils.detector import get_truck_detection
import cv2
import numpy as np
import os

path = 'C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 06/2021-05-22/'


def roadImage():
    # image = cv2.imread('C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 05/2021-05-08/2021-05-08T22_26_20.jpg')
    image = cv2.imread('C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 06/2021-05-20/2021-05-20T18_17_29.jpg')
    # image = cv2.imread('C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 07/2021-05-10/2021-05-11T07_08_42.jpg')
    return image   # [93:998, 625:1745]


execution_path = os.getcwd()
allData = ""

# truck_coord = {'Coordinates': [[0, 0, 0, 0], [0, 0, 0, 0]], 'TruckCount': 2}
# main_response = []
# truckCount = 0
# prevFileName, currFilename, area, Centroid, white_pixel_count

result = {
    0: {'Area': 00, 'Centroid': [0, 0], 'Coordinates': (0, 0, 0, 0), 'Image_name': ['default.jpg']}
}


# def whiteTHV():
#     return 10.5


# Working on previous frame
def checkID(med, area, x1, y1, x2, y2, prevFileName, currFilename, current_frame, previous_frame):
    global allData
    white = 0.0
    find = False
    count = 0
    for key, value in result.items():
        if (centTHV(value['Centroid'][0], '+') > med[0] > centTHV(value['Centroid'][0], '-')) and \
                (centTHV(value['Centroid'][1], '+') > med[1] > centTHV(value['Centroid'][1], '-')) and \
                (areaTHV(value['Area'], '+') > area > areaTHV(value['Area'], '-')):

            t = cv2.imread(path + str(value['Image_name'][-1]) + ".jpg")
            t = t[y1:y2, x1:x2]

            BSCount = backgroundSubtract(current_frame, t)
            print("Same Truck Detected of id, ", key)

            if BSCount['white_pixel_count'] < config['whiteTHV']:
                print("--Confirmed by Image Subtraction, Image name appended!")
                result[key]['Image_name'].append(currFilename) # Same Truck at same place on same day
                print('ID: ', key)
                find = True
                break
            else:
                print("--Truck has different features, not the same truck!")
        count += 1
    if find == False:
        print("Truck ID needed to be generated")

        Count = backgroundSubtract(current_frame, previous_frame)  # Just to confirm before appending
        white = Count["white_pixel_count"]

        if Count["white_pixel_count"] > config['whiteTHV']:
            # Confirm - Different Truck slided in
            print("--New Truck Confirmed by Background Subtraction")
            dict1 = {
                count: {'Area': area, 'Centroid': med, 'Coordinates': (x1, y1, x2, y2), 'Image_name': [currFilename]}}
            print('ID: ', count)
            result.update(dict1)
        else:
            print("--Same truck, Just calculation error! No new ID created")

    # allData += f"{prevFileName},{currFilename},{str(area(x1, y1, x2, y2))},{','.join(rectCentroid(x1, y1, x2, y2))},{str(white)},\n"
    # print(count)
    print("Updated Result Dict: ", result)
    # print("ALL DATA: \n", allData)


def area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


def rectCentroid(x1, y1, x2, y2):
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def centTHV(dictVal, sign):
    value = 15
    if sign == '-':
        return abs(dictVal - value)
    return dictVal + value


def areaTHV(dictVal, sign):
    value = 102000
    if sign == '-':
        return abs(dictVal - value)
    return dictVal + value


def cropImg(current_img, x1, y1, x2, y2):
    return current_img[y1:y2, x1:x2]


def backgroundSubtract(current_, previous_):
    global i
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
    tot_pix_count = h * w
    # print('Total Count of Pixel is', tot_pix_count)
    # Count = [np.sum(diff == 255), np.sum(diff == 0)]
    # {"black_pix_count": "", "white_pixel_count": ""}

    Count = {"black_pixel_count": float(np.sum(diff == 0) / tot_pix_count) * 100,
             "white_pixel_count": float(np.sum(diff == 255) / tot_pix_count) * 100}
    # print(Count)
    return Count


def count(diff):
    return np.sum(diff)


def TruckCount(Count):
    global truckCount
    if Count['white_pixel_count'] > config['whiteTHV']:
        truckCount += 1
    return truckCount


def firstImage(prev):
    startTime = time.time()
    ImageCopy = prev['frame'].copy()
    truck_coord_ = get_truck_detection(prev['frame'])

    # print("Time after detection- ", time.time()-startTime)

    # print(truck_coord_)

    global outputDir
    outputDir = os.path.join(execution_path, 'output', prev['fileName'].split('.')[0])

    if truck_coord_["Count"] <= 0:
        print("No trucks in first image")
        return
    # for multiple trucks detected on first image
    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords
        ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color2'], config['thickness'])
        # print(area(x1, y1, x2, y2))

        if area(x1, y1, x2, y2) > config['maxTruckSize']:
            ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])
            find = False
            count = 0
            for key, value in result.items():
                if (centTHV(value['Centroid'][0], '+') > rectCentroid(x1, y1, x2, y2)[0] > centTHV(value['Centroid'][0], '-'))\
                        and (centTHV(value['Centroid'][1], '+') > rectCentroid(x1, y1, x2, y2)[1] > centTHV(value['Centroid'][1], '-'))\
                        and (areaTHV(value['Area'], '+') > area(x1, y1, x2, y2) > areaTHV(value['Area'], '-')):
                    result[key]['Image_name'].append(prev['fileName'])
                    # value['Time'] += 15
                    print('ID: ', key)
                    find = True
                count += 1
            if find == False:
                print("Truck found on first Image")
                dict1 = {
                    count: {'Area': area(x1, y1, x2, y2), 'Centroid': rectCentroid(x1, y1, x2, y2), 'Coordinates': (x1, y1, x2, y2),
                            'Image_name': [prev['fileName']]}}
                print('ID: ', count)
                result.update(dict1)
            # allData += "NaN", prev['fileName'], area(x1, y1, x2, y2), rectCentroid(x1, y1, x2, y2), "NaN", "\n"
            print("Updated Result Dict: ", result)
    cv2.imwrite(str(outputDir) + '/Box.jpg', ImageCopy)
    # print("Time after Tracking- ", time.time()-startTime)

    return



noneDetectedImages = []


def Tracker(current_img, previous_img, actual):
    # global truckCount
    global outputDir
    outputDir = os.path.join(execution_path, 'output', current_img['fileName'].split('.')[0])
    # print("Current Image- ", outputDir)

    cv2.imwrite(str(outputDir) + '/Current.jpg', current_img['frame'])
    cv2.imwrite(outputDir + '/Previous.jpg', previous_img['frame'])
    # response = {current_img['fileName']: []}

    ImageCopy = current_img['frame'].copy()     # To draw bounding boxes
    # ImageCopy = cv2.rectangle(ImageCopy, (625, 93), (1745, 998), config['color3'], config['thickness'])
    ImageCopy = cv2.rectangle(ImageCopy, (0, 150), (1523, 1079), config['color3'], config['thickness'])

    # startTime = time.time()
    truck_coord_ = get_truck_detection(current_img['frame'])
    # print("Time after detection- ", time.time()-startTime)

    # print("Truck Coord from detector: ", truck_coord_)
    global i
    i = 0

    if truck_coord_["Count"] <= 0:
        print("No Truck Found")
        DCount = backgroundSubtract(current_img['frame'], previous_img['frame'])
        if DCount["white_pixel_count"] < config['whiteTHV']:
            print("Same as Previous Image")
        else:
            DCount = backgroundSubtract(current_img['frame'], roadImage())
            if DCount["white_pixel_count"] < config['whiteTHV']:
                print("No truck on road")
            else:
                for img in noneDetectedImages:
                    DCount = backgroundSubtract(current_img['frame'], img)
                    if DCount["white_pixel_count"] < config['whiteTHV']:
                        print("Non-detected Truck from some prev frame is repeated")
                        break
                noneDetectedImages.append(current_img['frame'])
                print("New image found & saved!")
        cv2.imwrite(execution_path + '/output' + '/ImagesCurrent' + '/box' + str(actual) + '.jpg', ImageCopy)
        return result

    # print(truck_coord_)
    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords

        ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color2'], config['thickness'])
        # print("Area: ", area(x1, y1, x2, y2))
        # print(rectCentroid(x1, y1, x2, y2))

        # if 625 < rectCentroid(x1, y1, x2, y2)[0] < 1745 and 93 < rectCentroid(x1, y1, x2, y2)[1] < 998:
        if 0 < rectCentroid(x1, y1, x2, y2)[0] < 1523 and 150 < rectCentroid(x1, y1, x2, y2)[1] < 1079:

            ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color3'], config['thickness'])
            if area(x1, y1, x2, y2) > config['maxTruckSize']:
                ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])

                current_frame = cropImg(current_img['frame'], x1, y1, x2, y2)
                previous_frame = cropImg(previous_img['frame'], x1, y1, x2, y2)
                cv2.imwrite(outputDir + '/CropCurrent' + str(i) + '.jpg', current_frame)
                cv2.imwrite(outputDir + '/CropPrevious' + str(i) + '.jpg', previous_frame)

                checkID(rectCentroid(x1, y1, x2, y2), area(x1, y1, x2, y2), x1, y1, x2, y2, previous_img['fileName'],
                        current_img['fileName'], current_frame, previous_frame)

                i += 1

        # if area(x1, y1, x2, y2) > config['maxTruckSize']:
        #     current_frame = cropImg(current_img['frame'], x1, y1, x2, y2)
        #     previous_frame = cropImg(previous_img['frame'], x1, y1, x2, y2)
        #
        #     cv2.imwrite(outputDir + '/CropCurrent' + str(i) + '.jpg', current_frame)
        #     cv2.imwrite(outputDir + '/CropPrevious' + str(i) + '.jpg', previous_frame)

        # cv2.imwrite('current.jpg', current_frame)
        # cv2.imwrite('previous.jpg', previous_frame)

        # Count = backgroundSubtract(current_frame, previous_frame)
        # truckCount = TruckCount(Count)

    #         i += 1
    #         individual_objects = {i: Count}
    #         response[current_img['fileName']].append(individual_objects)
    #         # truck_coord['Count'] -= 1
    # main_response.append(response)
    # print(main_response)
    # print("Day count of the Truck" + str(truckCount))
    cv2.imwrite(execution_path + '/output' + '/ImagesCurrent' + '/box' + str(actual) + '.jpg', ImageCopy)
    # cv2.imwrite(str(outputDir) + '/Box.jpg', ImageCopy)
    # print("Time after tracking- ", time.time()-startTime)

    return result

# main_data = {"image_current": [{id: {"black_pix_count": "", "white_pixel_count": ""}}, {id: {...}}..... ]
