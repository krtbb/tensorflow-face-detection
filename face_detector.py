#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

sys.path.append('/home/ec2-user/iTAC-FaceIdentifier-server/face_detection_tf')
from utils import label_map_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './face_detection_tf/model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './face_detection_tf/protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensorflowFaceDetector(object):
    def __init__(self, PATH_TO_CKPT, threshold=0.5):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        self.threshold = threshold
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_expanded = np.expand_dims(image, axis=0)
        #(boxes, scores, classes, num_detections) = self.sess.run(
        #    [boxes, scores, classes, num_detections],
        #    feed_dict={image_tensor: image_np_expanded})
        boxes, scores = self.sess.run( \
                [self.boxes, self.scores], \
                feed_dict = {self.image_tensor: image_expanded} \
                )
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        face_indices = np.where( scores > self.threshold )[0]
        face_boxes = boxes[face_indices]
        face_lefts = face_boxes[:, 1]
        face_boxes[:, 1] = face_boxes[:, 3]
        face_boxes[:, 3] = face_lefts

        return face_boxes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='path to image, "jpg" or "png"')
    args = parser.parse_args()
    
    tDetector = TensorflowFaceDetector(PATH_TO_CKPT)

    img_bgr = cv2.imread(args.image)
    boxes = tDetector.run(img_bgr)
    # "ymin, xmin, ymax, xmax" order
    h, w, c = img_bgr.shape
    for box in boxes:
        top, right, bottom, left = box
        top = int(top*h)
        bottom = int(bottom*h)
        left = int(left*w)
        right = int(right*w)
        cv2.rectangle(img_bgr, (left, top), (right, bottom), (155,155,155), 2)
        cv2.putText(img_bgr, str(score), (left,top), cv2.FONT_HERSHEY_PLAIN, 1, (155,155,155), 1, cv2.LINE_AA)
    output_image_path = args.image.replace('.', '_result.')
    cv2.imwrite(output_image_path, img_bgr)
    print('saved at {}'.format(output_image_path))
