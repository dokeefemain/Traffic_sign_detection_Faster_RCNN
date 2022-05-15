# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:07:26 2021

@author: Minas Benyamin

@editor: Devin O'Keefe
"""

import time
import cv2
import mss
import numpy as np
import tensorflow as tf
import pandas as pd
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from six import BytesIO
from PIL import Image
def annotate_sign(image, scoress, classess, boxess, width, height):
    t = time.time()
    global start_time
    for i in range(len(scoress)):
        class_name = category_index[classess[i]]['name']
        bbox = boxess[i]
        ymin, xmin, ymax, xmax = bbox
        (left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height))
        cv2.putText(img, class_name+str(round(scoress[i],2)), (left, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 50, 0), 1)
        cv2.rectangle(image, (left, top), (right, bottom), (255, 50, 0), 2)



def checkPoseDetection(image, frame_window, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    global frame_count
    tmp = time.time()

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        print(time.time()-tmp)
        scoress = np.squeeze(scores)
        classess = np.squeeze(classes).astype(np.int32)
        boxess = np.squeeze(boxes)

        test = np.where(scoress > 0.5)
        scores = scoress[test]
        classes = classess[test]
        boxes = boxess[test]

        #image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        width, height = image.shape[1], image.shape[0]
        image.flags.writeable = True

        annotate_sign(image, scores, classes, boxes, width, height)
    except Exception as e:
        print(e)
        image.flags.writeable = True
        left_location = int(50 * image.shape[0] / 480)
        font_scale = max(image.shape) / 800
        right_loc_1 = int(100 * image.shape[1] / 640)
        cv2.putText(image, "Looking for Target", (left_location, right_loc_1), cv2.FONT_HERSHEY_COMPLEX, font_scale,
                    (255, 50, 0), 2, lineType=cv2.LINE_AA)


    video_display(image)


def video_display(image):
    cv2.imshow('Faster RCNN IRV2', image)
    # cv2.imwrite('tmp/annotated_image' + str(counter) + '.png', image)
    vid_writer.write(image)


# For webcam input:
# cap = cv2.VideoCapture(input_source)
if __name__ == "__main__":

    input_source = 'garo_pullup_back.mov'
    output_source = 'tmp.mp4'
    start_time = time.time()
    curr_lab = ""
    curr_val = 0


    this_width = 800
    this_height = 500
    monitor = {'top': 200, 'left': 0, 'width': this_width, 'height': this_height}

    # cap = cv2.VideoCapture(0)
    # hasFrame, image = cap.read()


    labelmap_path = "lib/datasets/labelmap.pbtxt"
    path_to_ckpt = "lib/models/inference_graph/frozen_inference_graph.pb"
    num_classes = 46
    id = 36
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
    label_map = label_map_util.load_labelmap(labelmap_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    vid_writer = cv2.VideoWriter(output_source, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                                 (this_width, this_height))

    counter = 0
    condition = 1
    depth_condition = 1
    frame_count = 0
    bad_frames = 0

    var_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    old_x = var_x
    old_y = var_y
    frame_window = 10
    variance_collection = [0] * frame_window

    with mss.mss() as sct:
        while 'Screen capturing':
            frame_count = frame_count + 1
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            if True:
                checkPoseDetection(img, frame_window, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)
            else:
                print("Ignoring empty camera frame.")
                vid_writer.release()
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # hit escape to close video player
            if cv2.waitKey(5) & 0xFF == 27:
                vid_writer.release()
                break

        ratio = (bad_frames / frame_count) * 100
    print("Bad Frame Percentage: {:2.4}".format(ratio))
    if (ratio > 5):
        print('This video is not suitable for training the exercise prototypes at this time')

    cv2.destroyAllWindows()