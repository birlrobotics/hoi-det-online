# To avoid error if ROS is installed in the device
import sys
ROS_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)

import os
import cv2
import argparse
from PIL import Image
import numpy as np
from models.run import HOI_DET_ONLINE_MODEL

def hoi_det_online():
    # build and initialize the model
    model = HOI_DET_ONLINE_MODEL()

    if args.camera:
        # reading image with camera
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("Failed to open the video")
            sys.exit(1)

        while(1):
            ret, img = capture.read()
            if not ret and img in None:
                print("Miss a frames!!!")
                continue
            det_img = model(img, action_threshold=0.5, obj_det_only=args.obj_det_only)
            cv2.imshow('test', det_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()
    else:
        data_dir = '/home/birl/personal_data/bigjun/dataset/hico_20160224_det/images/test2015'
        img_list = os.listdir('./test_images/hico')
        for i in img_list:
            img = cv2.imread(os.path.join(data_dir, i))
            # pil_img = np.array(Image.open(os.path.join(data_dir, i)))
            # import ipdb; ipdb.set_trace()
            det_img = model(img, action_threshold=0.3, obj_det_only=args.obj_det_only)
            cv2.imshow('test', det_img)
            if cv2.waitKey(10000) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detecting HOIs in real!!!')
    parser.add_argument('--camera', action='store_true', help='use camera or not')
    parser.add_argument('--obj_det_only', action='store_true', help='just detect objects')
    args = parser.parse_args()

    hoi_det_online()
