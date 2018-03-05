# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

#This is negative_ignore version of imdb class for Caltech Pedestrian dataset


import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from utils import dataset_util

import json
from os.path import isfile, join, basename

import glob
#from datasets.config import CLASS_SETS
from natsort import natsorted







def create_tf_example(bboxes, img_info, category_name2id,class_mapper={}):
  # TODO(user): Populate the following variables from your example.
  

  height = img_info['height']
  width = img_info['width']
  filename = img_info['path']
  with tf.gfile.GFile(filename, 'rb') as fid:
    encoded_jpg = fid.read()
   
  image_format = img_info['format'] 
  
  

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
    
  for bbox in bboxes:
            
    xmin = float(bbox['x1']) / width
    xmax = float(bbox['x1'] + bbox['width']) / width
    ymin = float(bbox['y1']) / height 
    ymax = float(bbox['y1'] + bbox['height']) / height        
    class_text = class_mapper[bbox['label']] if bbox['label'] in class_mapper else bbox['label']
    if class_text == "__background__":
        continue
    class_id = category_name2id[class_text]

    xmins.append(xmin)
    xmaxs.append(xmax)
    ymins.append(ymin)
    ymaxs.append(ymax)    
    classes_text.append(str(class_text))        
    classes.append(class_id)

  
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(os.path.basename(filename)),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example




def get_data_map(path="/root/data", prefix="data-"):
    data_map = {} 
    data_paths = glob.glob("{}/{}*".format(path, prefix))
    for data_path in data_paths:
        name = basename(data_path)[5:]
        data_map[name] = data_path
    return data_map    

data_map = get_data_map()
data_names = data_map.keys()
    
    

def has_data(name):
    return name in data_names

def load_meta(meta_path):
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    else:
       
        meta = {"format":"jpg"}
        meta["train"] = {"start":None, "end":None, "stride":1, "sets":[0]}
        meta["test"] = {"start":None, "end":None, "stride":30, "sets":[1]}
        print("Meta data path: {} does not exist. Use Default meta data".format(meta_path))
    return meta
 

    
    
class IMDBGroup():
    
    
    
    
    def report(self):
        report = {}
        for vatic_data in self._datasets:
            for i in range(vatic_data.size):                
                for bbox in vatic_data.bbox_at(i):
                    label = bbox['label']
                    if label in vatic_data.CLS_mapper:
                        label = vatic_data.CLS_mapper[label]
                    report[label] = report.get(label, 0) + 1
            
        return report     
        

    
    def _get_img_paths(self):
        
        
        img_paths = []
        
        for dataset in self._datasets:
            for i in range(len(dataset._image_index)):
                img_path = dataset.image_path_at(i)
                img_paths.append(img_path)
            
        return img_paths   
            
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]
    
    
    
    def gt_roidb(self):
        
        gt_roidb = []
        for dataset in self._datasets:
            gt_roidb += dataset.gt_roidb()
        return gt_roidb

    
    
    def dumpTFRecord(self, output_path, category_name2id):
        writer = tf.python_io.TFRecordWriter(output_path)
        img_info = {}
        
        for vatic_data in self._datasets:
            

            if vatic_data.is_video:
                img = imread(vatic_data.image_path_at(0))
                img_info["height"] = img.shape[0]
                img_info['width'] = img.shape[1]
                img_info["format"] = vatic_data.image_path_at(0).split('.')[-1]

            for i in range(vatic_data.size):
                img_info['path'] = vatic_data.image_path_at(i)
                if not vatic_data.is_video:
                    img = imread(vatic_data.image_path_at(i))
                    img_info["height"] = img.shape[0]
                    img_info['width'] = img.shape[1]
                    img_info["format"] = vatic_data.image_path_at(i).split('.')[-1]

                bboxes = vatic_data.bbox_at(i)
                tf_record = create_tf_example(bboxes, img_info, category_name2id, vatic_data.CLS_mapper)
                writer.write(tf_record.SerializeToString())
        writer.close()  
        print("Tensorflow Record has been writen into {}".format(output_path))

            
            
            
    
    def __init__(self, datasets):
        self._datasets = datasets
        self.size = sum([dataset.size for dataset in datasets])
        
        
      

        
        
        
                


class VaticData():
    def __init__(self, name, train_split="train", test_split="test", CLS_mapper={}, is_video=True):
        
        assert data_map.has_key(name),\
        'The {} dataset does not exist. The available dataset are: {}'.format(name, data_map.keys())
            
        self._data_path = data_map[name]  
        assert os.path.exists(self._data_path), \
        'Path does not exist: {}'.format(self._data_path)
        
 
        
        annotation_path = os.path.join(self._data_path, "annotations.json")         
        assert os.path.exists(annotation_path), \
                'Annotation path does not exist.: {}'.format(annotation_path)
        self._annotation = json.load(open(annotation_path))   
        
        self.class_set = self.get_class_set()
        
        self.CLS_mapper = CLS_mapper
        self.is_video = is_video
        meta_data_path = os.path.join(self._data_path, "meta.json") 
        
       
            
        self._meta = load_meta(meta_data_path)
        
        if train_split == "train" or train_split ==  "test":
            pass
        elif train_split == "all":
            print("Use both split for training")
            self._meta["train"]["sets"] +=  self._meta["test"]["sets"]
        else:
            raise("Options except train and test are not supported!")
            
            
        if test_split == "train" or test_split ==  "test":
            pass
        elif test_split == "all":
            print("Use both split for testing")
            self._meta["test"]["sets"] +=  self._meta["train"]["sets"]
        else:
            raise("Options except train and test are not supported!")
        

        self._image_ext = self._meta["format"]    
        self._image_ext = '.jpg'
        self._image_index = self._get_image_index()
        self.size = len(self._image_index)
    
    
    
    
    
    def get_class_set(self):
        class_set = set()
        for set_num in self._annotation:
                for bboxes in self._annotation[set_num].values():
                    for bbox in bboxes.values():
                        class_set.add(bbox['label'])  
        
        
        
        return class_set
       
    
    def dumpTFRecord(self, output_path, category_name2id):
        writer = tf.python_io.TFRecordWriter(output_path)
        img_info = {}

        if self.is_video:
            img = imread(self.image_path_at(0))
            img_info["height"], img_info['width'], _ = img.shape
            img_info["format"] = self.image_path_at(0).split('.')[-1]
        
        for i in range(self.size):
            img_info['path'] = self.image_path_at(i)
            if not self.is_video:
                img = imread(self.image_path_at(i))
                img_info["height"], img_info['width'], _ = img.shape
                img_info["format"] = self.image_path_at(i).split('.')[-1]
                
            bboxes = self.bbox_at(i)
            tf_record = create_tf_example(bboxes, img_info, category_name2id, self.CLS_mapper)
            writer.write(tf_record.SerializeToString())
        writer.close()
        print("Tensorflow Record has been writen into {}".format(output_path))
            
        
        
            
    

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        set_num, v_num, frame = index.split("_")
        image_path = os.path.join(self._data_path, 'images', set_num, v_num, index+self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    
    def bbox_at(self, i):
        index = self._image_index[i]
        img_name = os.path.basename(index)
        set_num, v_num, img_num = img_name.split("_")
        set_num = set_num[-1]
       
        bboxes = self._annotation[set_num].get(img_num, {})
        return [bbox for bbox in bboxes.values() if bbox['outside']==0 and bbox['occluded']==0]
       
    
    
   
    def _load_image_set_list(self):
        image_set_file = os.path.join(self._data_path,
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        f = open(image_set_file)
        return  [line.strip() for line in f]
    
    
   
        

    def _get_image_index(self):
      
        """
        Load the indexes listed in this dataset's image set file.
        """
        
        
        image_path = os.path.join(self._data_path, 'images')
        assert os.path.exists( image_path), \
                'Path does not exist: {}'.format( image_path)
        target_imgs = []
        
        sets = self._meta["train"]["sets"]
        start = self._meta["train"]["start"]
        end = self._meta["train"]["end"]
        stride = self._meta["train"]["stride"]
        
        
        if start is None:
            start = 0
            
        for set_num in self._meta["train"]["sets"]:
            img_pattern = "{}/set0{}/V000/set0{}_V*.jpg".format(image_path,set_num,set_num)       
            img_paths = natsorted(glob.glob(img_pattern))
            #print(img_paths)
            
            first_ind = start
            last_ind = end if end else len(img_paths)
            for i in range(first_ind, last_ind, stride):
                img_path = img_paths[i]
                img_name = os.path.basename(img_path)
                target_imgs.append(img_name[:-4])
        print(self._meta)    
        print("Total: {} images".format(len(target_imgs)))            
        return target_imgs                  
      
    
#Unit Test
if __name__ == '__main__':
   
    
  
    name_a = "chruch_street"
    
    A = VaticData("chruch_street")
    B = VaticData("YuDa")
    imdb_group = IMDBGroup([A, B])
    imdb_group.dumpTFRecord("Test.record")
                         
    
    
    
    
    from IPython import embed; embed()