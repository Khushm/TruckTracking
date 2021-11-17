import shutil
import requests
from globals import config
import json
from utils.detector import get_truck_detection
import cv2
import numpy as np
import os
from loguru import logger
from mongoConn import get_mongo_client

execution_path = os.getcwd()
result = {}
currentPresigned_url = ""
previousPresigned_url = ""


def postRequest(outputDir, name):
    image_file = open(os.path.join(outputDir, name), "rb")
    image_file = image_file.read()
    mymetypes = "image/jpg"
    files = {'file': (name, image_file, mymetypes)}
    url = "http://api.smart-iam.com/api/image-store/upload/object/test"
    response = requests.post(url, files=files)
    response_json = json.loads(response.text)
    presigned_url = response_json['get_presignedUrl']
    return presigned_url


# Working on previous frame
def checkID(med, area, x1, y1, x2, y2, prevFileName, currFilename, current_frame, previous_frame, ImageCopy,
            prevImageCopy):
    global currentPresigned_url, previousPresigned_url
    global i
    white = 0.0
    find = False
    count = 0
    for key, value in result.items():
        if (centTHV(value['Centroid'][0], '+') > med[0] > centTHV(value['Centroid'][0], '-')) and \
                (centTHV(value['Centroid'][1], '+') > med[1] > centTHV(value['Centroid'][1], '-')) and \
                (areaTHV(value['Area'], '+') > area > areaTHV(value['Area'], '-')):
            t = cv2.imread(config['path'] + str(value['Image_name'][-1]) + ".jpg")
            t = t[y1:y2, x1:x2]

            BSCount = backgroundSubtract(current_frame, t)
            print("Same Truck Detected of id, ", key)

            if BSCount['white_pixel_count'] < config['whiteTHV']:
                print("--Confirmed by Image Subtraction, Image name appended!")
                result[key]['Image_name'].append(currFilename)  # Same Truck at same place on same day
                result[key]['Centroid'] = med
                result[key]['Area'] = area
                result[key]['Coordinates'] = (x1, y1, x2, y2)
                print('ID: ', key)

                ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])
                prevImageCopy = cv2.rectangle(prevImageCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])
                ImageCopy = cv2.putText(ImageCopy, key, (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                          config['color'], config['thickness'], cv2.LINE_AA)

                cv2.imwrite(outputDir + '/Current' + str(i) + currFilename + '.jpg', ImageCopy)
                cv2.imwrite(outputDir + '/Previous' + str(i) + prevFileName + '.jpg', prevImageCopy)

                currentPresigned_url = postRequest(outputDir, name='Current' + str(i) + currFilename + '.jpg')
                previousPresigned_url = postRequest(outputDir, name='Previous' + str(i) + prevFileName + '.jpg')
                return key
                # break
            else:
                # loop continues
                print("--Truck has different features, not the same truck!")
        count += 1
    if not find:
        print("Truck ID needed to be generated")
        Count = backgroundSubtract(current_frame, previous_frame)  # Just to confirm before appending

        if Count["white_pixel_count"] > config['whiteTHV']:
            # Confirm - Different Truck slided in
            print("--New Truck Confirmed by Background Subtraction")
            dict1 = {
                count: {'Area': area, 'Centroid': med, 'Coordinates': (x1, y1, x2, y2), 'Image_name': [currFilename]}}
            print('ID: ', count)
            result.update(dict1)
            return count
        else:
            print("--Same truck from previous frame, Just calculation error! No new ID created")
            return 0


def area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


def rectCentroid(x1, y1, x2, y2):
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def centTHV(dictVal, sign):
    if sign == '-':
        return abs(dictVal - config['centTHV'])
    return dictVal + config['centTHV']


def areaTHV(dictVal, sign):
    value = 102000
    if sign == '-':
        return abs(dictVal - config['areaTHV'])
    return dictVal + config['areaTHV']


def cropImg(current_img, x1, y1, x2, y2):
    return current_img[y1:y2, x1:x2]


def backgroundSubtract(current_, previous_, ):
    current = cv2.cvtColor(current_, cv2.COLOR_BGR2GRAY)
    previous = cv2.cvtColor(previous_, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(current, previous)

    _, diff = cv2.threshold(diff, 32, 0, cv2.THRESH_TOZERO)
    _, diff = cv2.threshold(diff, 62, 255, cv2.THRESH_BINARY)
    diff = cv2.medianBlur(diff, 5)

    h, w = diff.shape
    tot_pix_count = h * w

    Count = {"black_pixel_count": float(np.sum(diff == 0) / tot_pix_count) * 100,
             "white_pixel_count": float(np.sum(diff == 255) / tot_pix_count) * 100}
    return Count


def firstImage(prev):
    global outputDir
    outputDir = os.path.join(execution_path, 'output', prev['fileName'])

    # startTime = time.time()
    ImageCopy = prev['frame'].copy()
    ImageCopy = cv2.rectangle(ImageCopy, (625, 93), (1745, 998), config['color3'], config['thickness'])

    truck_coord_ = get_truck_detection(prev['frame'])
    # print("Time after detection- ", time.time()-startTime)

    if truck_coord_["Count"] <= 0:
        print("No trucks in first image")
        cv2.imwrite(outputDir + '/Current.jpg', ImageCopy)
        shutil.rmtree(outputDir)
        return

    # for multiple trucks detected on first image
    count = 1
    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords
        # ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color2'], config['thickness'])
        if config['minX'] < rectCentroid(x1, y1, x2, y2)[0] < config['maxX'] and \
                config['minY'] < rectCentroid(x1, y1, x2, y2)[1] < config['maxY']:
            if area(x1, y1, x2, y2) > config['maxTruckSize']:
                # ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])
                print("Truck found on first Image")
                dict1 = {
                    count: {'Area': area(x1, y1, x2, y2), 'Centroid': rectCentroid(x1, y1, x2, y2),
                            'Coordinates': (x1, y1, x2, y2),
                            'Image_name': [prev['fileName']]}}
                print('ID: ', count)
                result.update(dict1)

                currentCopy = ImageCopy.copy()
                currentCopy = cv2.rectangle(currentCopy, (x1, y1), (x2, y2), config['color'], config['thickness'])
                currentCopy = cv2.putText(currentCopy, count, (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                          config['color'], config['thickness'], cv2.LINE_AA)

                cv2.imwrite(outputDir + '/Current' + str(count) + '.jpg', currentCopy)
                currentPresigned_url = postRequest(outputDir, name='Current' + str(count) + '.jpg')

                count += 1
                Truckdata = {"TID": id, "Date": config['DateTime'].split("T")[0], "CAM": config["CAM"],
                             "ROI": (config["minX"], config["minY"], config["maxX"], config["maxY"]),
                             "CurrPresignedURL": currentPresigned_url,
                             "BoundingArea": (x1, y1, x2, y2),
                             "Time": (config['DateTime'].split("T")[1]).replace("_", ":")}
                print(Truckdata)
    shutil.rmtree(outputDir)


while True:
    mongo_coll = get_mongo_client()
    if mongo_coll:
        logger.info("Connected to MongoDB successfully.")
        break

noneDetectedImages = []


def Tracker(current_img, previous_img, actual):
    global outputDir, currentPresigned_url, previousPresigned_url

    outputDir = os.path.join(execution_path, 'output', current_img['fileName'])

    ImageCopy = current_img['frame'].copy()  # To draw bounding boxes
    prevImageCopy = previous_img['frame'].copy()

    ImageCopy = cv2.rectangle(ImageCopy, (config['minX'], config['minY']), (config['maxX'], config['maxY']), config['color3'], config['thickness'])
    prevImageCopy = cv2.rectangle(prevImageCopy, (config['minX'], config['minY']), (config['maxX'], config['maxY']), config['color3'], config['thickness'])

    # startTime = time.time()
    truck_coord_ = get_truck_detection(current_img['frame'])

    global i
    i = 0
    if truck_coord_["Count"] <= 0:
        print("No Truck Found")
        DCount = backgroundSubtract(current_img['frame'], previous_img['frame'])
        if DCount["white_pixel_count"] < config['whiteTHV']:
            print("Same as Previous Image")
        else:
            DCount = backgroundSubtract(current_img['frame'], config['roadImage'])
            if DCount["white_pixel_count"] < config['whiteTHV']:
                print("No truck on road")
        return result

    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords
        if config['minX'] < rectCentroid(x1, y1, x2, y2)[0] < config['maxX'] and config['minY'] < rectCentroid(x1, y1, x2, y2)[1] < config['maxY']:
            if area(x1, y1, x2, y2) > config['maxTruckSize']:
                current_frame = cropImg(current_img['frame'], x1, y1, x2, y2)
                previous_frame = cropImg(previous_img['frame'], x1, y1, x2, y2)

                id = checkID(rectCentroid(x1, y1, x2, y2), area(x1, y1, x2, y2), x1, y1, x2, y2,
                             previous_img['fileName'],
                             current_img['fileName'], current_frame, previous_frame, ImageCopy.copy(), prevImageCopy.copy())

                Truckdata = {"TID": id, "Date": current_img['fileName'].split("T")[0], "CAM": config["CAM"],
                             "ROI": (config["minX"], config["minY"], config["maxX"], config["maxY"]),
                             "CurrPresignedURL": currentPresigned_url, "PrevPresignedURL": previousPresigned_url,
                             "BoundingArea": (x1, y1, x2, y2),
                             "Time": (current_img['fileName'].split("T")[1]).replace("_", ":")
                             }

                print(Truckdata)
                post_id = mongo_coll.insert_one(Truckdata).inserted_id
                print(post_id)
        i += 1
    print("Result", result)
    shutil.rmtree(outputDir)



# Constraints of Project:
#     First occurrence of the truck must be detected else the truck would never get an ID
#     If area centroid doesn't match and bgSubtraction error is less than threshold then error in passing the truck ID
#     Day-Night lightening condition error


# Solution
#     Improve Truck detection algorithm
#     Maintain dict of all truck objects detected and match with them, instead of just storing previous image values
#     None
