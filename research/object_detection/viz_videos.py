import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
import time
sys.path.append("..")  #Not sure if I need this line if I run it in the parental directory
from utils import label_map_util
import glob
import cv2
from matplotlib import pyplot as plt
from skvideo.io import FFmpegWriter
from nms.cpu_nms import cpu_nms as nms

MODELS = ['finetune_01', 'faster_rcnn_inception_v2_coco_2017_11_08']
NUM_CLASSES = 90
PATH_TO_LABELS = "data/mscoco_label_map.pbtxt"
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_model(model, dynamic_memory=True):


    #Dynamically allocating memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=dynamic_memory
    sess = tf.Session(config=config)    
    PATH_TO_CKPT = os.path.join(model , 'frozen_inference_graph.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return tf.Session(graph=detection_graph, config=config)





def translate_result(boxes, scores, classes, num_detections, im_width, im_height, thresh):
    #Normalizing the detection result
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)    
    
    thresh_mask = scores > thresh
    
    scores = scores[thresh_mask]
    boxes = boxes[thresh_mask]
    classes = classes[thresh_mask]
    
    outputs = []        
    for i, score in enumerate(scores):      
       
    
        class_name = category_index[classes[i]]['name']
        ymin, xmin, ymax, xmax = boxes[i]
        left, right, top, bottom = (xmin * im_width, xmax * im_width,\
                                  ymin * im_height, ymax * im_height)          
        #Allocating result into ouput dict
        output = {}      

        output['score'] = score
        output['class'] = class_name
        output['x'] = left
        output['y'] = top
        output['width'] = right-left
        output['height'] = bottom-top
        #Append each detection into a list
        print(output)
        outputs.append(output)
    return outputs


def translate_result_NMS(boxes, scores, classes, im_width, im_height, thresh, NMS_THRESH=0.5, TYPE_NUM=90):
    #Normalizing the detection result
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)    
    
    thresh_mask = scores > thresh
    
    scores = scores[thresh_mask]
    boxes = boxes[thresh_mask]
    classes = classes[thresh_mask].astype(int)
    
    outputs = []   
    
    for cls_ind in range(TYPE_NUM):
        cls_mask = (classes == cls_ind)
        if np.sum(cls_mask) == 0:           
            continue
        else:
            print(category_index[cls_ind]['name'])

        filtered_boxes = boxes[cls_mask]
        filtered_scores = scores[cls_mask]

        dets = np.hstack((boxes[cls_mask], scores[cls_mask][:, np.newaxis]))
        keep = nms(dets, NMS_THRESH)

        filtered_boxes = filtered_boxes[keep]
        filtered_scores = filtered_scores[keep]


        for i, score in enumerate(filtered_scores):      
        

            class_name = category_index[cls_ind]['name']
            ymin, xmin, ymax, xmax = filtered_boxes[i]
            left, right, top, bottom = (xmin * im_width, xmax * im_width,\
                                      ymin * im_height, ymax * im_height)          
            #Allocating result into ouput dict
            output = {}      

            output['score'] = score
            output['class'] = class_name
            output['x'] = left
            output['y'] = top
            output['width'] = right-left
            output['height'] = bottom-top
            #Append each detection into a list
            print(output)
            outputs.append(output)
    return outputs

def detect_img(sess, img, thresh=0.7):
    #img = Image.open(img_path)
    #
    #img_np = load_image_into_numpy_array(img)
  
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
    img_height, img_width, _ = img.shape
    img_np_expanded = np.expand_dims(img, axis=0)
    
    #Initalization of output and input tensors for session
    img_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    scores = sess.graph.get_tensor_by_name('detection_scores:0')
    classes = sess.graph.get_tensor_by_name('detection_classes:0')
    num_detections = sess.graph.get_tensor_by_name('num_detections:0')
    
    outputs = [boxes, scores, classes, num_detections]
    feed_dict = {img_tensor: img_np_expanded}
    boxes, scores, classes, num_detections = sess.run(outputs,feed_dict=feed_dict) 
 
    return translate_result_NMS(boxes, scores, classes, img_width,\
                            img_height,thresh)







def render_frame(im, dts):
    for dt in dts:
        xmin = int(dt["x"])
        ymin = int(dt["y"])
        width = int(dt["width"])
        height = int(dt["height"])
        label = dt['class']
        score = int(dt['score'] * 100) 
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, "{}-{}".format(label, score), (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im
        
        
def write_video(input_path, output_path):
    i = 0
    print("Reading Video {}".format(input_path))
    input_video = cv2.VideoCapture(input_path)
    print("Reading Finished")
    output_video = FFmpegWriter(output_path)
    while(True):
        ret, input_frame = input_video.read()
        if input_frame is None:
            break
        print(input_frame.shape)
        detections = detect_img(sess, input_frame)
        output_frame = render_frame(input_frame, detections)
        output_video.writeFrame(output_frame)
        i += 1
        print("Writen Frame: {}".format(i))
    output_video.close()
    
    
font = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.8



if __name__ == '__main__':
    INPUT_PATH = "/root/test.AVI"
    OUTPUT_PATH = "/root/result.mp4"
    model = MODELS[1]
    sess = load_model(model)

    write_video(INPUT_PATH, OUTPUT_PATH)