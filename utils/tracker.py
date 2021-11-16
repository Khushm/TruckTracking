import datetime
import shutil
import time

import requests

from globals import config
import json
from utils.detector import get_truck_detection
import cv2
import numpy as np
import os
from loguru import logger
from mongoConn import get_mongo_client

path = 'C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 05/2021-05-07/'

prevImageDetectionCount = 0


def roadImage():
    image = cv2.imread('C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 05/2021-05-08/2021-05-08T22_26_20.jpg')
    # image = cv2.imread('C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 06/2021-05-20/2021-05-20T18_17_29.jpg')
    # image = cv2.imread('C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 07/2021-05-11/2021-05-11T07_08_42.jpg')
    return image  # [93:998, 625:1745]


execution_path = os.getcwd()
allData = ""
ImageID = {}
# truck_coord = {'Coordinates': [[0, 0, 0, 0], [0, 0, 0, 0]], 'TruckCount': 2}
# main_response = []
# truckCount = 0
# prevFileName, currFilename, area, Centroid, white_pixel_count

result = {
    0: {'Area': 00, 'Centroid': [0, 0], 'Coordinates': (0, 0, 0, 0), 'Image_name': ['default.jpg']}
}


def postRequest(outputDir, name):
    global presigned_url, object_url
    # f = open(os.path.join(outputDir, name), "rb")
    # image_file = cv2.imread(os.path.join(outputDir, name))

    image_file = open(os.path.join(outputDir, name), "rb")
    image_file = image_file.read()

    mymetypes = "image/jpg"
    files = {'file': (name, image_file, mymetypes)}

    url = "http://api.smart-iam.com/api/image-store/upload/object/test"

    response = requests.post(url, files=files)
    response_json = json.loads(response.text)

    presigned_url = response_json['get_presignedUrl']
    # object_url = response_json['get_object']


# Working on previous frame
def checkID(med, area, x1, y1, x2, y2, prevFileName, currFilename, current_frame, previous_frame, ImageCopy,
            prevImageCopy):
    global allData, prevImageDetectionCount, presigned_url, object_url, currentPresigned_url, previousPresigned_url
    global i
    white = 0.0
    find = False
    count = 0
    for key, value in result.items():
        if (centTHV(value['Centroid'][0], '+') > med[0] > centTHV(value['Centroid'][0], '-')) and \
                (centTHV(value['Centroid'][1], '+') > med[1] > centTHV(value['Centroid'][1], '-')) and \
                (areaTHV(value['Area'], '+') > area > areaTHV(value['Area'], '-')):
            currentCopy = ImageCopy.copy()
            previousCopy = prevImageCopy.copy()
            currentCopy = cv2.rectangle(currentCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])
            previousCopy = cv2.rectangle(previousCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])

            cv2.imwrite(outputDir + '/Current' + str(i) + currFilename + '.jpg', currentCopy)
            cv2.imwrite(outputDir + '/Previous' + str(i) + prevFileName + '.jpg', previousCopy)

            postRequest(outputDir, name='Current' + str(i) + currFilename + '.jpg')
            currentPresigned_url = presigned_url

            # print("C", currentPresigned_url)

            postRequest(outputDir, name='Previous' + str(i) + prevFileName + '.jpg')
            previousPresigned_url = presigned_url
            # a = os.path.join(outputDir, 'Current' + str(i) + currFilename + '.jpg')
            # print("Remove",a)
            # os.remove(os.path.join(outputDir, 'Current' + str(i) + currFilename + '.jpg'))
            # os.remove(os.path.join(outputDir, 'Previous' + str(i) + prevFileName + '.jpg'))

            # print("P", previousPresigned_url, previousObject_url)

            t = cv2.imread(path + str(value['Image_name'][-1]) + ".jpg")
            t = t[y1:y2, x1:x2]

            BSCount = backgroundSubtract(current_frame, t)
            # dict1["diff"] = BSCount["white_pixel_count"]

            print("Same Truck Detected of id, ", key)

            if BSCount['white_pixel_count'] < config['whiteTHV']:
                print("--Confirmed by Image Subtraction, Image name appended!")
                result[key]['Image_name'].append(currFilename)  # Same Truck at same place on same day
                print('ID: ', key)
                find = True

                # ImageID.append({currFilename['fileName']: key})
                return key
                # break
            else:
                print("--Truck has different features, not the same truck!")
        count += 1
    if find == False:
        print("Truck ID needed to be generated")

        Count = backgroundSubtract(current_frame, previous_frame)  # Just to confirm before appending
        # dict1["diff"] = Count["white_pixel_count"]

        # white = Count["white_pixel_count"]

        if Count["white_pixel_count"] > config['whiteTHV']:
            # Confirm - Different Truck slided in
            print("--New Truck Confirmed by Background Subtraction")
            dict1 = {
                count: {'Area': area, 'Centroid': med, 'Coordinates': (x1, y1, x2, y2), 'Image_name': [currFilename]}}
            print('ID: ', count)
            result.update(dict1)
            return count
        else:
            print("--Same truck, Just calculation error! No new ID created")
            return count - 1

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


def backgroundSubtract(current_, previous_, ):
    global i
    current = cv2.cvtColor(current_, cv2.COLOR_BGR2GRAY)
    previous = cv2.cvtColor(previous_, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(current, previous)

    _, diff = cv2.threshold(diff, 32, 0, cv2.THRESH_TOZERO)
    _, diff = cv2.threshold(diff, 62, 255, cv2.THRESH_BINARY)
    diff = cv2.medianBlur(diff, 5)
    # cv2.imwrite(outputDir + '/Diff' + str(i) + '.jpg', diff)

    h, w = diff.shape
    tot_pix_count = h * w

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
    # startTime = time.time()
    ImageCopy = prev['frame'].copy()
    ImageCopy = cv2.rectangle(ImageCopy, (625, 93), (1745, 998), config['color3'], config['thickness'])

    truck_coord_ = get_truck_detection(prev['frame'])

    # print("Time after detection- ", time.time()-startTime)

    # print(truck_coord_)

    global outputDir, prevImageDetectionCount, presigned_url, object_url, currentPresigned_url
    outputDir = os.path.join(execution_path, 'output', prev['fileName'].split('.')[0])

    if truck_coord_["Count"] <= 0:
        prevImageDetectionCount = 0
        print("No trucks in first image")
        cv2.imwrite(outputDir + '/Current.jpg', ImageCopy)
        # os.remove(os.path.join())
        # shutil.rmtree(outputDir)
        return

    # for multiple trucks detected on first image
    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords
        # ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color2'], config['thickness'])
        # print(area(x1, y1, x2, y2))
        if 625 < rectCentroid(x1, y1, x2, y2)[0] < 1745 and 93 < rectCentroid(x1, y1, x2, y2)[1] < 998:
            if area(x1, y1, x2, y2) > config['maxTruckSize']:
                currentCopy = ImageCopy.copy()
                currentCopy = cv2.rectangle(currentCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])
                cv2.imwrite(outputDir + '/Current' + str(i) + '.jpg', currentCopy)

                postRequest(name=outputDir + '/Current' + str(i) + '.jpg')
                currentPresigned_url = presigned_url

                prevImageDetectionCount += 1
                # ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])
                find = False
                count = 0
                for key, value in result.items():
                    if (centTHV(value['Centroid'][0], '+') > rectCentroid(x1, y1, x2, y2)[0] > centTHV(
                            value['Centroid'][0], '-')) \
                            and (centTHV(value['Centroid'][1], '+') > rectCentroid(x1, y1, x2, y2)[1] > centTHV(
                        value['Centroid'][1], '-')) \
                            and (areaTHV(value['Area'], '+') > area(x1, y1, x2, y2) > areaTHV(value['Area'], '-')):
                        result[key]['Image_name'].append(prev['fileName'])
                        # value['Time'] += 15
                        print('ID: ', key)
                        find = True
                    count += 1
                if find == False:
                    print("Truck found on first Image")
                    dict1 = {
                        count: {'Area': area(x1, y1, x2, y2), 'Centroid': rectCentroid(x1, y1, x2, y2),
                                'Coordinates': (x1, y1, x2, y2),
                                'Image_name': [prev['fileName']]}}
                    print('ID: ', count)
                    result.update(dict1)
                # allData += "NaN", prev['fileName'], area(x1, y1, x2, y2), rectCentroid(x1, y1, x2, y2), "NaN", "\n"

                Truckdata = {"ID": id, "Date": config['DateTime'].split("T")[0], "CAM": config["CAM"],
                             "ROI": (config["minX"], config["minY"], config["maxX"], config["maxY"]),
                             "CurrPresignedURL": currentPresigned_url,
                             "BoundingArea": (x1, y1, x2, y2),
                             "Time": (config['DateTime'].split("T")[1]).replace("_", ":")}

                print(Truckdata)
                print("Updated Result Dict: ", result)
    # cv2.imwrite(str(outputDir) + '/Box.jpg', ImageCopy)
    # print("Time after Tracking- ", time.time()-startTime)
    # shutil.rmtree(outputDir)
    return


while True:
    mongo_coll = get_mongo_client()
    if mongo_coll:
        logger.info("Connected to MongoDB successfully.")
        break

noneDetectedImages = []


def write_json(new_data, filename='data.json'):
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data["TruckTrack"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

currentPresigned_url = ""
previousPresigned_url = ""

def Tracker(current_img, previous_img, actual):
    dict = {"prevImage": previous_img['fileName'], "currImage": current_img['fileName'], "detection": []}

    # global truckCount
    global outputDir, prevImageDetectionCount, currentPresigned_url, previousPresigned_url

    outputDir = os.path.join(execution_path, 'output', current_img['fileName'].split('.')[0])
    # print("Current Image- ", outputDir)

    # cv2.imwrite(str(outputDir) + '/Current.jpg', current_img['frame'])
    # cv2.imwrite(outputDir + '/Previous.jpg', previous_img['frame'])
    # response = {current_img['fileName']: []}

    ImageCopy = current_img['frame'].copy()  # To draw bounding boxes
    prevImageCopy = previous_img['frame'].copy()

    ImageCopy = cv2.rectangle(ImageCopy, (625, 93), (1745, 998), config['color3'], config['thickness'])
    prevImageCopy = cv2.rectangle(prevImageCopy, (625, 93), (1745, 998), config['color3'], config['thickness'])

    # ImageCopy = cv2.rectangle(ImageCopy, (0, 150), (1523, 1079), config['color3'], config['thickness'])
    # ImageCopy = cv2.rectangle(ImageCopy, (config['minX'], config['minY']), (config['maxX'], config['maxY']), config['color3'], config['thickness'])
    # startTime = time.time()
    truck_coord_ = get_truck_detection(current_img['frame'])
    # print("Time after detection- ", time.time()-startTime)

    # print("Truck Coord from detector: ", truck_coord_)
    global i
    i = 0
    currentDetectionCount = 0
    if truck_coord_["Count"] <= 0:
        print("No Truck Found")
        DCount = backgroundSubtract(current_img['frame'], previous_img['frame'])
        if DCount["white_pixel_count"] < config['whiteTHV']:
            print("Same as Previous Image")
        else:
            DCount = backgroundSubtract(current_img['frame'], config['roadImage'])
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
        # cv2.imwrite(execution_path + '/output' + '/ImagesCurrent' + '/box' + str(actual) + '.jpg', ImageCopy)
        cv2.imwrite(str(outputDir) + '/Current.jpg', ImageCopy)
        cv2.imwrite(outputDir + '/Previous.jpg', prevImageCopy)
        shutil.rmtree(outputDir)
        return result

    # print(truck_coord_)
    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords
        # dict1 = {"prevBB": (x1, y1, x2, y2), "currBB": (x1, y1, x2, y2), "diff": ""}
        # ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color2'], config['thickness'])
        # print("Area: ", area(x1, y1, x2, y2))
        # print(rectCentroid(x1, y1, x2, y2))

        if 625 < rectCentroid(x1, y1, x2, y2)[0] < 1745 and 93 < rectCentroid(x1, y1, x2, y2)[1] < 998:
            # if 0 < rectCentroid(x1, y1, x2, y2)[0] < 1523 and 150 < rectCentroid(x1, y1, x2, y2)[1] < 1079:

            # if config['minX'] < rectCentroid(x1,y1,x2,y2)[0] < config['maxX'] and config['minY'] < rectCentroid(x1,y1,x2,y2)[1] < config['maxY']:
            #     ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color3'], config['thickness'])
            if area(x1, y1, x2, y2) > config['maxTruckSize']:
                current_frame = cropImg(current_img['frame'], x1, y1, x2, y2)
                previous_frame = cropImg(previous_img['frame'], x1, y1, x2, y2)
                # cv2.imwrite(outputDir + '/CropCurrent' + str(i) + '.jpg', current_frame)
                # cv2.imwrite(outputDir + '/CropPrevious' + str(i) + '.jpg', previous_frame)
                i += 1
                id = checkID(rectCentroid(x1, y1, x2, y2), area(x1, y1, x2, y2), x1, y1, x2, y2,
                             previous_img['fileName'],
                             current_img['fileName'], current_frame, previous_frame, ImageCopy, prevImageCopy)
                #
                Truckdata = {"ID": id, "Date": config['DateTime'].split("T")[0], "CAM": config["CAM"],
                             "ROI": (config["minX"], config["minY"], config["maxX"], config["maxY"]),
                             "CurrPresignedURL": currentPresigned_url, "PrevPresignedURL": previousPresigned_url,
                             "BoundingArea": (x1, y1, x2, y2),
                             "Time": (config['DateTime'].split("T")[1]).replace("_", ":")
                }
                #
                print(Truckdata)
                print("Result", result)
                post_id = mongo_coll.insert_one(Truckdata).inserted_id
                print(post_id)
                # dict["detection"].append(dict1)

    # main_response.append(response)
    # print(main_response)
    # print("Day count of the Truck" + str(truckCount))
    cv2.imwrite(execution_path + '/output' + '/ImagesCurrent' + '/box' + str(actual) + '.jpg', ImageCopy)
    # cv2.imwrite(str(outputDir) + '/Box.jpg', ImageCopy)
    # print("Time after tracking- ", time.time()-startTime)

    # finalData.append(dict)
    # print("Formatted Data List: ", finalData)
    # write_json(finalData)
    shutil.rmtree(outputDir)
    return result
