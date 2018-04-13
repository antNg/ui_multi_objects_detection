import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2


class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def normalize(image):
    image = image / 255.
    
    return image

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2
    
    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2
    
    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    
    intersect = intersect_w * intersect_h
    
    union = box1.w * box1.h + box2.w * box2.h - intersect
    
    return float(intersect) / union
    
def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def draw_boxes(image, boxes, labels, colors=None, line_type=cv2.LINE_4, thickness=1):
    """
        Many part of this function adapted from (https://github.com/experiencor/basic-yolo-keras)
        Modified to add color palette.
        Author:
            Nguyen Thanh An <annt@vng.com.vn>
    """
    default_color = (0, 255, 0)
    colors = {
        'person':   (240, 163, 10),
        'car':      (0, 255, 255),
        'chair':    (0, 220, 0),
        'book':     (0, 255, 255),
        'bottle':   (250, 104, 0),
        'cup':      (47, 141, 255),
        'dining table': (216, 0, 115),
        'bowl':     (255, 224, 32),
        'traffic light': (229, 20, 0),
        'handbag':  (0, 255, 255),
        'umbrella': (118, 57, 49),
        'boat': (100, 209, 62),
        'bird': (174, 142, 94),
        'truck': (0, 38, 133),
        'banana': (0, 126, 58),
        'bench': (0, 38, 133),
        'sheep': (76, 14, 119),
        'kite': (205, 30, 16),
        'backpack': (100, 209, 62),
        'motorcycle': (118, 57, 49),
        'potted plant': (250, 223, 0),
        'cow': (66, 154, 223),
        'carrot': (241, 171, 0),
        'knife': (252, 0, 127),
        'bicycle': (77, 199, 253),
        'skis': (0, 38, 133),
        'vase': (250, 223, 0),
        'horse': (205, 30, 16),
        'orange': (241, 171, 0),
        'cake': (0, 38, 133),
        'cell phone': (118, 57, 49),
        'tie': (118, 57, 49),
        'sports ball': (77, 199, 253),
        'bus': (76, 14, 119),
        'clock': (250, 223, 0),
        'apple': (205, 30, 16),
        'spoon': (250, 223, 0),
        'suitcase': (0, 38, 133),
        'surfboard': (0, 126, 58),
        'couch': (100, 209, 62),
        'tv': (77, 199, 253),
        'skateboard': (118, 57, 49),
        'sink': (77, 199, 253),
        'elephant': (118, 57, 49),
        'airplane': (66, 154, 223),
        'dog': (77, 199, 253),
        'zebra': (250, 223, 0),
        'giraffe': (94, 83, 199),
        'laptop': (0, 38, 133),
        'cat': (254, 121, 209),
        'bed': (66, 154, 223),
        'toilet': (0, 126, 58),
        'keyboard': (241, 171, 0),
        'refrigerator': (66, 154, 223),
        'frisbee': (205, 30, 16),
        'mouse': (94, 83, 199),
        'toothbrush': (77, 199, 253),
        'stop sign': (241, 171, 0),
        'fire hydrant': (66, 154, 223),
        'microwave': (126, 119, 210),
        'bear': (118, 57, 49),
        'parking meter': (76, 14, 119)
    }
    colors = {k: tuple(reversed(v)) for k, v in colors.items()}
    text_color = (255, 255, 255)
    
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        label_name = labels[box.get_label()]
        color = colors.get(label_name) if colors.get(label_name) else default_color
        label_name = 'table' if label_name == 'dining table' else label_name
        text_pos = (xmin + 5, ymin - 6)
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_value = '{}'.format(label_name, box.get_score())
        font_scale = 0.4

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness, line_type)

        cv2.rectangle(image, (xmin, ymin - 20), (xmin + 100, ymin), color, -1)
        cv2.putText(image, 
                    text_value,
                    text_pos,
                    text_font,
                    font_scale, 
                    text_color,
                    1,
                    line_type)
        
    return image        
        
def decode_netout(netout, obj_threshold, nms_threshold, anchors, nb_class):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x, y, w, h, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)