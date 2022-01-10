import logger
import numpy as np
import cv2 as cv
import time
import os
import json


def plot_tracking(image, tlwhs, obj_ids, frame_id=0, fps=0., ids2=None):
    try:
        im = np.ascontiguousarray(np.copy(image))
        im_h, im_w = im.shape[:2]

        top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

        #text_scale = max(1, image.shape[1] / 1600.)
        #text_thickness = 2
        #line_thickness = max(1, int(image.shape[1] / 500.))
        text_scale = 1
        text_thickness = 2
        line_thickness = 1

        radius = max(5, int(im_w/140.))
        cv.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                   (0, int(15 * text_scale)), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=1)

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1+w, y1+h)))
            obj_id = int(obj_ids[i])
            id_text = '{}'.format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))
            cv.rectangle(im, intbox[0:2], intbox[2:4],
                         color=(255, 0, 0), thickness=line_thickness)
            cv.putText(im, id_text, (intbox[0], intbox[1]), cv.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 0),
                       thickness=text_thickness)
            # cv.imwrite(f'frame_{id_text}_{frame_id}.jpg', im)
            # os.remove(f'frame_{self._id}_{self.frame_id}.jpg')
        return id_text, im
    except Exception as e:
        logger.error("Error in plot_tracking | {}".format(e))


def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)


def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    try:
        # If there are any detections
        if len(idxs) > 0:
            for i in idxs.flatten():
                # global count = 0
                if labels[classids[i]] == 'person':
                    # count+=1
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]

                    # Get the unique color for this class
                    color = [int(c) for c in colors[classids[i]]]

                    # Draw the bounding box rectangle and label on the image
                    cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    cv.circle(img, (int(x+(w/2)), int(y+(h/2))), 5, (0, 255, 0), 2)
                    text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
                    cv.putText(img, text, (x, y-5),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img, boxes, confidences
    except Exception as e:
        logger.error("Error in draw_labels_and_boxes | {}".format(e))


def generate_boxes_confidences_classids(outs, height, width, tconf):
    try:
        boxes = []
        confidences = []
        classids = []

        for out in outs:
            for detection in out:
                #print (detection)
                #a = input('GO!')

                # Get the scores, classid, and the confidence of the prediction
                scores = detection[5:]
                classid = np.argmax(scores)
                confidence = scores[classid]

                # Consider only the predictions that are above a certain confidence level
                if confidence > tconf:

                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, bwidth, bheight = box.astype('int')

                    # Using the center x, y coordinates to derive the top
                    # and the left corner of the bounding box
                    x = int(centerX - (bwidth / 2))
                    y = int(centerY - (bheight / 2))

                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence))
                    classids.append(classid)

        return boxes, confidences, classids
    except Exception as e:
        logger.error("Error in generate_boxes_confidences_classids | {}".format(e))


def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS=None,
                boxes=None, confidences=None, classids=None, idxs=None, infer=True, draw_box=False):
    try:
        if infer:
            show_time = False
            confidence = 0.5
            threshold = 0.2
            # Contructing a blob from the input image
            blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                        swapRB=True, crop=False)

            # Perform a forward pass of the YOLO object detector
            net.setInput(blob)

            # Getting the outputs from the output layers
            start = time.time()
            outs = net.forward(layer_names)
            end = time.time()

            if show_time:
                print("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

            # Generate the boxes, confidences, and classIDs
            boxes, confidences, classids = generate_boxes_confidences_classids(
                outs, height, width, confidence)

            # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
            idxs = cv.dnn.NMSBoxes(
                boxes, confidences, confidence, threshold)

        if boxes is None or confidences is None or idxs is None or classids is None:
            raise '[ERROR] Required variables are set to None before drawing boxes on images.'

        # Draw labels and boxes on the image
        if draw_box:
            img_, boxes_, confidences_ = draw_labels_and_boxes(
                img, boxes, confidences, classids, idxs, colors, labels)
            return img_, boxes, confidences, classids, idxs

        # dk = dlib_tracker(img_, boxes_, confidences_)

        return img, boxes, confidences, classids, idxs
    except Exception as e:
        logger.error("Error in infer_image | {}".format(e))
