from utils.detector import get_truck_detection
import cv2
import numpy as np
import os

execution_path = os.getcwd()

truck_coord = {'Coordinates': [[0, 0, 0, 0], [0, 0, 0, 0]], 'TruckCount': 2}
main_response = []

# truck_coord = {x1: 0,y1: 0, x2: 0, y2: 0, 'Count': 0}

def Canny_detector(img, weak_th=None, strong_th=None):
    # conversion of image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction step
    img = cv2.GaussianBlur(img, (5, 5), 1.4)

    # Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # Conversion of Cartesian coordinates to polar
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # setting the minimum and maximum thresholds
    # for double thresholding
    mag_max = np.max(mag)
    if not weak_th: weak_th = mag_max * 0.1
    if not strong_th: strong_th = mag_max * 0.5

    # getting the dimensions of the input image
    height, width = img.shape

    # Looping through every pixel of the grayscale
    # image
    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # top right (diagonal-1) direction
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            # In y-axis direction
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

            # top left (diagonal-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1

            # Now it restarts the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Non-maximum suppression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue

            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)
    ids = np.zeros_like(img)

    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):

            grad_mag = mag[i_y, i_x]

            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2

    # finally returning the magnitude of
    # gradients of edges
    return mag


def area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


def maxTruckSize():
    value = 65000
    return value


def cropImg(current_img, x1, y1, x2, y2):
    return current_img[y1:y2, x1:x2]


def backgroundSubtract(current, previous):

    # current = Canny_detector(current)
    # previous = Canny_detector(previous)

    current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('curr.jpg', current)
    cv2.imwrite('prev.jpg', previous)

    diff = cv2.absdiff(current, previous)
    _, diff = cv2.threshold(diff, 32, 0, cv2.THRESH_TOZERO)
    _, diff = cv2.threshold(diff, 62, 255, cv2.THRESH_BINARY)
    diff = cv2.medianBlur(diff, 5)
    cv2.imwrite('diff.jpg', diff)
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

    response = {current_img['fileName']: []}

    truck_coord_ = get_truck_detection(previous_img['frame'])
    i = 0;
    if len(truck_coord_["Co_Ordinates"]) <= 0:
        print("No detections Fpund!")
        return main_response
    print(truck_coord_)
    for cords in truck_coord_['Co_Ordinates']:
        x1, y1, x2, y2 = cords

        if area(x1, y1, x2, y2) > maxTruckSize():
            current_frame = cropImg(current_img['frame'], x1, y1, x2, y2)
            previous_frame = cropImg(previous_img['frame'], x1, y1, x2, y2)

            cv2.imwrite('current.jpg', current_frame)
            cv2.imwrite('previous.jpg', previous_frame)


            Count = backgroundSubtract(current_frame, previous_frame)
            i += 1
            individual_objects = {i: Count}
            response[current_img['fileName']].append(individual_objects)
            # truck_coord['Count'] -= 1

    main_response.append(response)
    print(main_response)
    return main_response


# main_data = {"image_current": [{id: {"black_pix_count": "", "white_pixel_count": ""}}, {id: {...}}..... ]