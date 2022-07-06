"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import re
import time
import argparse
import datetime
import threading

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps, return_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


WINDOW_NAME = 'TrtYOLODemo'
cam_port = 0
FilePath = "jetson2.jpg"
start_time = 0
end_time = 0

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args

def image_detection():
    os.system("python3 trt_yolo_image.py -m yolov4-tiny-416 --image jetson2.jpg")

def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    global start_time, end_time
    full_scrn = False
    fps = 0.0
    tic = time.time()
    
    jetson2_fps = open("jetson2_fps.txt", "r")
    jetson2_other_fps = open("jetson2_other_fps.txt", "w")
    jetson2_other_detail = open("jetson2_other_detail.txt", "w")
    
    jetson1_fps = open("jetson1_fps.txt", "a+")
    jetson1_detail = open("jetson1_detail.txt", "w")
    
    print("Executing obejet detection now...")
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()

        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        
        jetson1_fps.write(str(return_fps(fps)) + "\n")
        jetson1_detail.write("jetson1 " + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "\n")
        jetson1_detail.write(str(confs) + "\n")
        jetson1_detail.write(str(clss) + "\n")

        jetson1_fps.flush()
        jetson1_detail.flush()
        
        now_time = datetime.datetime.now().strftime("%S")

        if (int(str(now_time)) % 5 == 0):
            jetson1_fps.seek(0, 0)
            jetson1_total_fps = jetson2_fps.read()
            jetson1_sum_fps = [float(i) for i in re.findall(r'[\d\.\d]+', jetson1_total_fps)]
            jetson1_line_fps = sum(1 for line in open("jetson1_fps.txt"))
            jetson1_average_fps = float(f'{sum(jetson1_sum_fps)}') / jetson1_line_fps

        if (int(str(now_time)) % 5 == 0):
            os.system("./receive_jetson1.sh")

        #img = vis.draw_bboxes(img, boxes, confs, clss)
        #img = show_fps(img, fps)
        #cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)


        if (int(str(now_time)) % 5 == 0):
            os.system("./send_jetson1.sh")
            jetson2_fps.seek(0, 0)
            jetson2_total_fps = jetson1_fps.read()
            jetson2_sum_fps = [float(i) for i in re.findall(r'[\d\.\d]+', jetson2_total_fps)]
            
            jetson2_line_fps = sum(1 for line in open("jetson2_fps.txt"))
            if (float(f'{sum(jetson2_sum_fps)}') != 0 and jetson2_line_fps != 0 and float(f'{sum(jetson1_sum_fps)}') != 0 and jetson1_line_fps != 0):
                jetson2_average_fps = float(f'{sum(jetson2_sum_fps)}') / jetson2_line_fps
            
                cam = cv2.VideoCapture(cam_port)
            
                print("1: "+ str(jetson1_average_fps))
                print("2: "+ str(jetson2_average_fps))
            
                if (jetson1_average_fps < jetson2_average_fps):
                    print("Using Jetson 2")
                    cv2.imwrite("jetson1.jpg", img)
                    os.system("./send_jetson1_image.sh")
                else:
                    print("Using Jetson 1")
                    os.system("./receive_jetson2_image.sh")
                    start_time = time.localtime(os.stat(FilePath).st_ctime)

                    if(start_time != end_time):
                        thread = threading.Thread(target = image_detection)
                        thread.start()
                        print("Thread start")

                    end_time = time.localtime(os.stat(FilePath).st_ctime)
                    thread.join()
                    print("Thread end")

        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
