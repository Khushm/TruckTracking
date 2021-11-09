from utils.tracker import Tracker, firstImage
# from utils.writedata import makeSheet
import cv2
import os

execution_path = os.getcwd()
path = 'C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 05/2021-05-06'

# makeSheet(path.split('/')[-1])

image_path = os.listdir(path)

# previous_img = {'frame': cv2.imread(os.path.join(execution_path, "images", "21.jpg")), 'fileName': "21.jpg"}
i = 1

outputDir = os.path.join(execution_path, 'output')
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

for each_files in image_path:
    print(each_files)
    fileName = str(each_files.split('.')[0])
    if i == 1:
        # print("in")
        previousImg = {'frame': cv2.imread(os.path.join(path, each_files)), 'fileName': fileName}
        firstImage(previousImg)
        i += 1
        print('------------------------------------')
        continue

    filePath = os.path.join(outputDir, fileName)
    if not os.path.exists(filePath):
        os.mkdir(filePath)
        # print("done")

    currentImg = {'frame': cv2.imread(os.path.join(path, each_files)), 'fileName': fileName}

    Tracker(currentImg, previousImg)
    print('------------------------------------')
    previousImg = currentImg