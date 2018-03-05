import skvideo.io
from skvideo.io import FFmpegWriter
import cv2
import numpy as np


    
if __name__ == '__main__':
    model_A = "faster_rcnn_inception_v2_coco_2017_11_08"
    model_B = "rc1_180k"
    video_names = ["bike2.mp4", "bus_motorbike.mp4", "cat_dog.mp4", "suitcase_person.mp4", "truck_car.mp4"]
    for video_name in video_names[1:]:
        
        cap = cv2.VideoCapture
        input_path_A = "/root/data/testing_videos/res/{}_{}".format(model_A, video_name)
        input_path_B = "/root/data/testing_videos/res/{}_{}".format(model_B, video_name)
        input_video_A = cv2.VideoCapture(input_path_A)
        input_video_B = cv2.VideoCapture(input_path_B)
        

        OUTPUT_PATH = "/root/data/testing_videos/res/{}_{}".format("compare", video_name)       
        output_video = FFmpegWriter(OUTPUT_PATH)
        while True:
            _, input_frame_A = input_video_A.read()
            if input_frame_A is None:
                break
            _, input_frame_B = input_video_B.read()
            
        #for i, input_frame_A in enumerate(input_video_A):
            #input_frame_B = input_video_B[i]
            height, width, _ = input_frame_A.shape
            output_frame = np.zeros((height*2, width, 3), dtype=np.uint8)
            output_frame[:height,:,:] = cv2.cvtColor(input_frame_A, cv2.COLOR_BGR2RGB)
            output_frame[height:,:,:] = cv2.cvtColor(input_frame_B, cv2.COLOR_BGR2RGB)
            output_video.writeFrame(output_frame)
        output_video.close()    
            
        print(OUTPUT_PATH) 

        
