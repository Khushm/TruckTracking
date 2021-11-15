from utils.tracker import Tracker, firstImage
from globals import config
import json
import cv2
import os
# finalDict = {"PrevLink": previous_img['fileName'], "CurrLink": current_img['fileName'],
                             # "BBox": (x1, y1, x2, y2)}

execution_path = os.getcwd()
file = open('read.json')
data = json.load(file)
outputDir = os.path.join(execution_path, 'output')
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

for i in data['TruckTrack']:
    config['CAM'] = i["camName"]
    fromT = i["fromTime"]
    toT = i['toTime']
    config["minX"], config["minY"], config["maxX"], config["maxY"] = [int(x) for x in i['ROI'].split(",")]
    config['DateTime'] = i["fromTime"]
    path = os.path.join('C:/Users/munda/PycharmProjects/IAmSmart-T0/images/', config['CAM'], fromT.split("T")[0])
    image_path = os.listdir(path)


    i = 0
    for each_files in image_path:
        each_files_ = each_files.split(".")[0]
        if each_files_ > toT:
            break

        if fromT < each_files_ or each_files_ == toT:
            print(each_files)
            fileName = str(each_files.split('.')[0])
            i += 1

            if i == 1:
                previousImg = {'frame': cv2.imread(os.path.join(path, each_files)), 'fileName': fileName}
                firstImage(previousImg)
                print('------------------------------------')
                continue

            filePath = os.path.join(outputDir, fileName)
            if not os.path.exists(filePath):
                os.mkdir(filePath)

            currentImg = {'frame': cv2.imread(os.path.join(path, each_files)), 'fileName': fileName}
            Tracker(currentImg, previousImg, i)
            print('------------------------------------')
            previousImg = currentImg
        #
        # else:
        #     print("Not in given range")
    # finalData.clear()

file.close()

