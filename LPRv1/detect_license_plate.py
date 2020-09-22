import sys
import numpy as np
import os
import tensorflow as tf
import cv2


class CarNumberPlateDetectorv1(object):
    def __init__(self):
        super(CarNumberPlateDetectorv1, self).__init__()
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                                'model/frozen_inference_graph.pb'), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                self.sess = tf.Session()

    def detect(self, image_np):
        (im_height, im_width) = image_np.shape[:2]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
          [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
          feed_dict={self.image_tensor: image_np_expanded})
        bbs = []
        for box in boxes:
            ymin = box[0,0]
            xmin = box[0,1]
            ymax = box[0,2]
            xmax = box[0,3]
            (xmin, xmax, ymin, ymax) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            (xmin, xmax, ymin, ymax) = max(0, int(xmin)), min(im_width, int(xmax)), max(0, int(ymin)), min(im_height, int(ymax))
            bbs.append([xmin, ymin, xmax, ymax])
        return bbs