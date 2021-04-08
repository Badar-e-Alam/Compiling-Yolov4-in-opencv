# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np

def predection(image):
    # image = cv2.imread(filename=image_name)
    print(cv2.__version__)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    # read class names from text file
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    # read pre-trained model and config file
    net = cv2.dnn.readNet("yolov4-csp.weights", "yolov4.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)    # create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    outs = net.forward(output_layers)

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.1)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print(type(x))
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 5)
            cv2.putText(image, label, (int(x), int(y) + 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 5)

    # display output image
    return image

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    net= cv2.dnn.readNet("yolov4-csp.weights", "yolov4-x.cfg")


    #
    # # cv2.imshow('img',predection(cv2.imread('img.png')))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
