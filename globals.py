# road Image
# CAM 05 - C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 05/2021-05-08/2021-05-08T22_26_20.jpg
# CAM 06 - C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 06/2021-05-20/2021-05-20T18_17_29.jpg
# CAM 07 - C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 07/2021-05-11/2021-05-11T07_08_42.jpg

# path
# CAM 05 - C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 05/2021-05-06/
# CAM 06 - C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 06/2021-05-06/
# CAM 07 - C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 07/2021-05-06/

# ROI
# CAM 05 [625, 93] [1745, 998]
# CAM 06 [0, 150] [1523, 1079]
# CAM 07 [769, 38] [1891, 1079]

import cv2
config = {
    "thickness": 2,
    "color3": (0, 255, 0),  # Green
    "color2": (0, 0, 255),  # Red
    "color": (255, 0, 0),   # Blue

    "whiteTHV": 10.5,
    "maxTruckSize": 90000,
    "centTHV": 15,
    "areaTHV": 102000,

    "roadImage": cv2.imread('C:/Users/munda/PycharmProjects/IAmSmart-T0/images/CAM 07/2021-05-11/2021-05-11T07_08_42.jpg'),
}
