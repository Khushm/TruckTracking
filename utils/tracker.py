import time

from utils.detector import get_truck_detection
import cv2
import numpy as np
import os

color2 = (0, 0, 255)
color = (255, 0, 0)
thickness = 2
path = 'C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 05/2021-05-07/'


def roadImage():
    image = cv2.imread(path + '2021-05-06T23_45_24.jpg')
    return image[93:998, 625:1745]


execution_path = os.getcwd()
allData = ""

# truck_coord = {'Coordinates': [[0, 0, 0, 0], [0, 0, 0, 0]], 'TruckCount': 2}
# main_response = []
# truckCount = 0
# prevFileName, currFilename, area, Centroid, white_pixel_count

result = {
    0: {'Area': 00, 'Centroid': [0, 0], 'Coordinates': (0, 0, 0, 0), 'Image_name': ['default.jpg']}
}


def whiteTHV():
    return 10.5


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
            t = t[93:998, 625:1745]
            # tx1, ty1, tx2, ty2 = value['Coordinates']
            #
            # # if t.shape != current_frame.shape:
            # print("t", tx1, ty1, tx2, ty2)
            print("c", x1, y1, x2, y2)
            # print("Not equal")

            # ax1 = x1 if x1 > tx1 else tx1
            # ax2 = x2 if x2 < tx2 else tx2
            # ay1 = y1 if y1 > ty1 else ty1
            # ay2 = y2 if y2 < ty2 else ty2
            # print("actual", ax1, ay1, ax2, ay2)

            # t = t[ay1:ay2, ax1:ax2]
            # current_frame = current_frame[ay1:ay2, ax1:ax2]
            # cv2.imshow("T frame", t)
            # cv2.imshow("Current Frame", current_frame)
            # cv2.waitKey(5)

            # if y2-y1 > ty2-ty1:
            #     if x2-x1 > tx2-tx1:
            #         print("1")
            #         current_frame = current_frame[ty1:ty2, tx1:tx2]
            #     else:
            #         print("2")
            #         current_frame = current_frame[ty1:ty2, x1:x2]
            #         t = t[ty1:ty2, x1:x2]
            # else:
            #     if x2 - x1 > tx2 - tx1:
            #         print("3")
            #         current_frame = current_frame[y1:y2, tx1:tx2]
            #         t = t[y1:y2, tx1:tx2]
            #     else:
            #         print("4")
            print("before", t.shape, current_frame.shape)

            t = t[y1:y2, x1:x2]

                # if t.shape < current_frame.shape:
                #     print("c", tx1, ty1, tx2, ty2)
                #     current_frame = current_frame[ty1:ty2, tx1:tx2]
                #
                # else:
                #     t = t[y1:y2, x1:x2]
            print("s", t.shape, current_frame.shape)
            BSCount = backgroundSubtract(current_frame, t)
            print("Same Truck Detected of id, ", key)

            if BSCount['white_pixel_count'] < whiteTHV():
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

        # if previous_frame.shape != current_frame.shape:
        #     if previous_frame.shape < current_frame.shape:
        #         gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        #         edges = cv2.Canny(gray, 50, 200)
        #         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #         sorted_contours = sorted(contours, key=cv2.contoursArea, reverse=False)
        #         largest_item = sorted_contours[-1]
        #         x, y, w, h = cv2.boundingRect(largest_item)
        #         current_frame = current_frame[y:y + h, x:x + w]
        #     else:
        #         previous_frame = previous_frame[y1:y2, x1:x2]

        Count = backgroundSubtract(current_frame, previous_frame)  # Just to confirm before appending
        white = Count["white_pixel_count"]

        if Count["white_pixel_count"] > whiteTHV():
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


def maxTruckSize():
    value = 107501   #120000
    return value


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
    print(Count)
    return Count


def count(diff):
    return np.sum(diff)


def TruckCount(Count):
    global truckCount
    if Count['white_pixel_count'] > whiteTHV():
        truckCount += 1
    return truckCount


def firstImage(prev):
    startTime = time.time()
    ImageCopy = prev['frame'].copy()
    truck_coord_ = get_truck_detection(prev['frame'])

    print("Time after detection- ", time.time()-startTime)

    print()
    # print(truck_coord_)

    global outputDir
    outputDir = os.path.join(execution_path, 'output', prev['fileName'].split('.')[0])

    if truck_coord_["Count"] <= 0:
        print("No trucks in first image")
        return
    # for multiple trucks detected on first image
    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords
        ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), color2, thickness)
        print(area(x1, y1, x2, y2))

        if area(x1, y1, x2, y2) > maxTruckSize():
            ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), color, thickness)
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
    print("Time after Tracking- ", time.time()-startTime)

    return



noneDetectedImages = []


def Tracker(current_img, previous_img):
    # global truckCount
    global outputDir
    outputDir = os.path.join(execution_path, 'output', current_img['fileName'].split('.')[0])
    # print("Current Image- ", outputDir)

    cv2.imwrite(str(outputDir) + '/Current.jpg', current_img['frame'])
    cv2.imwrite(outputDir + '/Previous.jpg', previous_img['frame'])
    # response = {current_img['fileName']: []}

    ImageCopy = current_img['frame'].copy()     # To draw bounding boxes

    startTime = time.time()
    truck_coord_ = get_truck_detection(current_img['frame'])
    print("Time after detection- ", time.time()-startTime)

    # print("Truck Coord from detector: ", truck_coord_)
    global i
    i = 0

    if truck_coord_["Count"] <= 0:
        print("No Truck Found")
        DCount = backgroundSubtract(current_img['frame'], previous_img['frame'])
        if DCount["white_pixel_count"] < whiteTHV():
            print("Same as Previous Image")
        else:
            DCount = backgroundSubtract(current_img['frame'], roadImage())
            if DCount["white_pixel_count"] < whiteTHV():
                print("No truck on road")
            else:
                for img in noneDetectedImages:
                    DCount = backgroundSubtract(current_img['frame'], img)
                    if DCount["white_pixel_count"] < whiteTHV():
                        print("Non-detected Truck from some prev frame is repeated")
                        break
                noneDetectedImages.append(current_img['frame'])
                print("New image found & saved!")
        return result

    # print(truck_coord_)
    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords
        ImageCopy = cv2.rectangle(ImageCopy, (x1,y1), (x2,y2), color2, thickness)

        if area(x1, y1, x2, y2) > maxTruckSize():
            ImageCopy = cv2.rectangle(ImageCopy, (x1, y1), (x2, y2), color, thickness)

            current_frame = cropImg(current_img['frame'], x1, y1, x2, y2)
            previous_frame = cropImg(previous_img['frame'], x1, y1, x2, y2)
            cv2.imwrite(outputDir + '/CropCurrent' + str(i) + '.jpg', current_frame)
            cv2.imwrite(outputDir + '/CropPrevious' + str(i) + '.jpg', previous_frame)

            checkID(rectCentroid(x1, y1, x2, y2), area(x1, y1, x2, y2), x1, y1, x2, y2, previous_img['fileName'],
                    current_img['fileName'], current_frame, previous_frame)

            i += 1

        # if area(x1, y1, x2, y2) > maxTruckSize():
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
    cv2.imwrite(str(outputDir) + '/Box.jpg', ImageCopy)
    print("Time after tracking- ", time.time()-startTime)

    return result

# main_data = {"image_current": [{id: {"black_pix_count": "", "white_pixel_count": ""}}, {id: {...}}..... ]
