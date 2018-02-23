train_dir = "/root/models/research/object_detection/rc1/"
config_path = "/root/models/research/object_detection/config/AI_VMS_RC1_fastrrcnn_inception.config"
GPU_ID = 3


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
from train import FLAGS, main, tf


if __name__ == '__main__':
    FLAGS.train_dir = train_dir
    FLAGS.pipeline_config_path = config_path    
    
    main(FLAGS)