import skvideo.io
from skvideo.io import FFmpegWriter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys, cv2
import glob
from nms.py_cpu_nms import py_cpu_nms as nms
from utils import label_map_util

GPU_ID = 3
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
import tensorflow as tf

MODELS = ['output_inference_graph.pb', 'faster_rcnn_inception_v2_coco_2017_11_08', 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28/', 'rc1_180k', 'faster_rcnn_nas_coco_2018_01_28']
#model = MODELS[1]
model = MODELS[4]
NUM_CLASSES = 90
#NUM_CLASSES = 9
PATH_TO_LABELS = "/root/models/research/object_detection/data/mscoco_label_map.pbtxt"
#PATH_TO_LABELS = "/root/models/research/object_detection/data/AI_VMS_label_map.pbtxt"
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
font = cv2.FONT_HERSHEY_DUPLEX
font_size = 0.8
NMS_THRESH = 0.5
TARGET_LABELS = ["bus", "motorbike", "car", "bicycle", "person", "dog", "cat", "truck", "suitcase", "motorcycle"]















def detect_img(sess, img, thresh=0.7, NMS_THRESH=0.7):
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
   
    
    outputs = [boxes, scores, classes]
    feed_dict = {img_tensor: img_np_expanded}
    boxes, scores, classes = sess.run(outputs,feed_dict=feed_dict) 
    print(np.max(scores))
 
    return translate_result_NMS(boxes, scores, classes, img_width,\
                            img_height,thresh, NMS_THRESH=NMS_THRESH)


#Get the session from the specified model
def load_model(model, dynamic_memory=True):


    #Dynamically allocating memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=dynamic_memory
    sess = tf.Session(config=config)    
    PATH_TO_CKPT = os.path.join("/root/models/research/object_detection", model , 'frozen_inference_graph.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return tf.Session(graph=detection_graph, config=config)



def translate_result_NMS(boxes, scores, classes, im_width, im_height, thresh, roi=(0,0,0,0), NMS_THRESH=0.7, TYPE_NUM=90):
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
        if np.sum(cls_mask) <= 1:
            continue           
        filtered_boxes = boxes[cls_mask]
        filtered_scores = scores[cls_mask]
        dets = np.hstack((filtered_boxes,  filtered_scores[:, np.newaxis]))
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
            output['x'] = left + roi[0]
            output['y'] = top + roi[1]
            output['width'] = right-left
            output['height'] = bottom-top
            #Append each detection into a list
            if class_name in TARGET_LABELS:
                outputs.append(output)
    return outputs




def render_frame(im, dts):
    #Drawing Title
    if TITLE is not None:
        cv2.putText(im, TITLE, (15,50), font, font_size*2, (255,255,255),3)   
    for dt in dts:
        xmin = int(dt["x"])
        ymin = int(dt["y"])
        width = int(dt["width"])
        height = int(dt["height"])
        label = dt['class']
        highlight_W = xmin + len(label) * 14
        highlight_H = ymin + height
        cv2.rectangle(im, (xmin,ymin),(xmin+width,ymin+height),(0,255,0),2)
        cv2.rectangle(im, (xmin,ymin+height+14),(highlight_W, highlight_H),(0,255,0),-1)
        cv2.putText(im, label, (xmin, highlight_H+14), font, font_size, (0,0,0),1)       
    return im
        
        
def write_video(input_path, output_path, sess):
    i = 0
    print("Reading Video {}".format(input_path))
    input_video = skvideo.io.vread(input_path)
    print("Reading Finished")
    output_video = FFmpegWriter(output_path)
    for input_frame in input_video:
        print(input_frame.shape)
        dts = detect_img(sess, input_frame, NMS_THRESH=NMS_THRESH)
        output_frame = render_frame(input_frame, dts)
        output_video.writeFrame(output_frame)
        i += 1
        print("Writen Frame: {}".format(i))
    output_video.close()

    
    
if __name__ == '__main__':
    TITLE=model
    sess = load_model(model)
    video_names = ["bike2.mp4", "bus_motorbike.mp4", "cat_dog.mp4", "suitcase_person.mp4", "truck_car.mp4"]
    for video_name in video_names:
        INPUT_PATH = "/root/data/testing_videos/{}".format(video_name)
        OUTPUT_PATH = "/root/data/testing_videos/res/{}_{}".format(model, video_name)       
        print(OUTPUT_PATH) 
        write_video(INPUT_PATH, OUTPUT_PATH, sess)



