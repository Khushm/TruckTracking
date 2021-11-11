from utils.tracker import Tracker, firstImage
# from utils.writedata import makeSheet
import cv2
import os

execution_path = os.getcwd()
path = 'C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 05/2021-05-20'

# makeSheet(path.split('/')[-1])

image_path = os.listdir(path)

# previous_img = {'frame': cv2.imread(os.path.join(execution_path, "images", "21.jpg")), 'fileName': "21.jpg"}
i = 0

outputDir = os.path.join(execution_path, 'output')
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

for each_files in image_path:
    print(each_files)
    fileName = str(each_files.split('.')[0])
    i += 1
    if i == 1:
        # print("in")
        previousImg = {'frame': cv2.imread(os.path.join(path, each_files)), 'fileName': fileName}
        # cv2.imwrite(execution_path + '/output' + '/ImagesCurrent' + '/Current' + str(i) + '.jpg' , previousImg['frame'])
        firstImage(previousImg)
        print('------------------------------------')
        continue

    filePath = os.path.join(outputDir, fileName)
    if not os.path.exists(filePath):
        os.mkdir(filePath)
        # print("done")

    currentImg = {'frame': cv2.imread(os.path.join(path, each_files)), 'fileName': fileName}
    # print(currentImg['frame'].shape)

    # cv2.imwrite(execution_path + '/output' + '/ImagesCurrent' + '/Current' + str(i) + '.jpg', currentImg['frame'])
    # [628,31], [1881,1046] = x1,y1,x2,y2 - CAM 5 - 2021-05-06
    # [633,81]
    # [601,11]
    # [625, 93], [1745,998]

    Tracker(currentImg, previousImg, i)
    print('------------------------------------')
    previousImg = currentImg