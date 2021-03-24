# To avoid error if ROS is installed in the device
import sys
ROS_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)

import os
import cv2
import time
import argparse
from PIL import Image
import numpy as np
from models.run import HOI_DET_ONLINE_MODEL

def hoi_det_online():
    # # build and initialize the model
    model = HOI_DET_ONLINE_MODEL(obj_det_only=args.obj_det_only, use_pmn=args.use_pmn)
    # get the numbers of parameters of the designed model
    param_dict = {}
    for param in model.named_parameters():
        moduler_name = param[0].split('.')[0]
        if moduler_name in param_dict.keys():
            param_dict[moduler_name] += param[1].numel()
        else:
            param_dict[moduler_name] = param[1].numel()
    for k, v in param_dict.items():
        print(f"{k} Parameters: {v / 1e6} million.")
    print(f"Parameters in total: {sum(param_dict.values()) / 1e6} million.")

    if args.camera:
        key = str(time.time()).split('.')[-1]
        if args.save_img:
            if not os.path.exists(f'./results/original_img/{key}'):
                os.makedirs(f'./results/original_img/{key}')
            if not os.path.exists(f'./results/processed_img/{key}'):
                os.makedirs(f'./results/processed_img/{key}')
        # reading image with camera
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            print("Failed to open the video")
            sys.exit(1)
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video = cv2.VideoWriter(f'./results/camera_test_{key}.avi', fourcc, 5.0, size)
        cv2.namedWindow('camera', 0)
        cv2.resizeWindow('camera', 1280, 960)
        n = 1
        mit = 0
        while(1):
            ret, img = capture.read()
            if not ret and img in None:
                print("Miss a frames!!!")
                continue
            t1 = time.time()
            det_img = model(img, action_threshold=args.act_threshold, show_line=args.show_line, show_pose=args.show_pose)
            t2 = time.time()
            mit = (mit*(n-1)+(t2-t1)) / n
            print(f"Moving mean inference time: {mit}s.")
            cv2.imshow('camera', det_img)
            
            if args.save_img:
                # cv2.imwrite(f'./results/original_img/{key}/{n}.png', img)
                cv2.imwrite(f'./results/processed_img/{key}/{n}.png', det_img)
            if args.save_video:
                video.write(det_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            n+=1
        capture.release()
        cv2.destroyAllWindows()
    else:
        data_dir = '/home/birl/personal_data/bigjun/dataset/hico_20160224_det/images/test2015'
        img_list = os.listdir('./test_images/hico')
        cv2.namedWindow('offline', 0)
        for i in img_list:
            # import ipdb; ipdb.set_trace()
            img = cv2.imread(os.path.join(data_dir, i))
            det_img = model(img, action_threshold=args.act_threshold, show_line=args.show_line, show_pose=args.show_pose)
            cv2.imshow('offline', det_img)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detecting HOIs in real!!!')
    parser.add_argument('--camera', action='store_true', help='use camera or not')
    parser.add_argument('--obj_det_only', action='store_true', help='just detect objects')
    parser.add_argument('--use_pmn', action='store_true', help='use PMN model or not')
    parser.add_argument('--save_img', action='store_true', help='save the image')
    parser.add_argument('--save_video', action='store_true', help='record the visualization as video')
    parser.add_argument('--show_line', action='store_true', help='visulaize the line connecting the human and object')
    parser.add_argument('--show_pose', action='store_true', help='visulaize the detected human pose')
    parser.add_argument('--act_threshold', type=float, default=0.5, help='action threshold')
    args = parser.parse_args()

    hoi_det_online()


# import torchvision
# import torch
# from utils import metadata, vis_tool

        # model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, box_batch_size_per_image=128, \
        #                                                                box_score_thresh=0.3, box_nms_thresh=0.3)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device).eval()

        # # model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, box_batch_size_per_image=128, \
        # #                                                                box_score_thresh=0.3, box_nms_thresh=0.3)
        # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # # model.to(device).eval()
        # data_dir = '/home/birl/personal_data/bigjun/dataset/hico_20160224_det/images/test2015'
        # img_list = os.listdir('./test_images/hico')
        # cv2.namedWindow('offline', 0)
        # for i in img_list:
        #     # import ipdb; ipdb.set_trace()
        #     img = cv2.imread(os.path.join(data_dir, i))
        #     # pil_img = np.array(Image.open(os.path.join(data_dir, i)))
        #     # img = img[:,:,::-1].copy()
        #     # img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)
        #     # output = model([img_tensor])
        #     # obj_boxes = output[1][0]['boxes']
        #     # obj_labels = output[1][0]['labels']
        #     # pose = output[1][0]['keypoints']
        #     # det_img = vis_tool.vis_img_frcnn(img, obj_boxes.cpu().detach().numpy(), obj_labels.cpu().detach().numpy(), pose=pose.cpu().detach().cpu().numpy(), score_thresh=0.8)
        #     # det_img = np.array(det_img)[:,:,::-1]
        #     det_img = model(img, action_threshold=args.act_threshold, show_line=args.show_line, show_pose=args.show_pose)
        #     cv2.imshow('offline', det_img)
        #     if cv2.waitKey(1000) & 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()