import random
import numpy as np
from utils import metadata
from PIL import Image, ImageDraw, ImageFont
import cv2

POSE_LINKED_LIST = {0: [1,2,5,6],
                    1: [3],
                    2: [4],
                    3: None,
                    4: None,
                    5: [6,7,11],
                    6: [8,12],
                    7: [9],
                    8: [10],
                    9: None,
                    10: None,
                    11: [12,13],
                    12: [14],
                    13: [15],
                    14: [16],
                    15: None,
                    16: None
                    }

def vis_img(img, bboxs, labels, scores, pose=None, act_prob=None, score_thresh=0.5, show_line=True, show_pose=False):
    try:
        if len(bboxs) == 0:
            return img    

        human_num = len(np.where(labels == 1)[0])
        node_num = len(labels)
        labeled_edge_list = np.cumsum(node_num - np.arange(human_num) -1)
        labeled_edge_list[-1] = 0
        h_mark = {idx:0 for idx in range(human_num)}
        color_mark = {idx:None for idx in range(node_num)}
        pose_mark = {idx:0 for idx in range(human_num)}
        det_hoi_num = 0

        line_thickness = 2
        # https://blog.csdn.net/u011520181/article/details/84110517
        fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontScale = 1
        font_thickness = 1

        for h_idx in range(human_num):
            for i_idx in range(node_num):
                if i_idx <= h_idx:
                    continue
                edge_idx = labeled_edge_list[h_idx-1] + (i_idx-h_idx-1)
                # if h_idx == i_idx:
                #     continue
                # if h_idx > i_idx:
                #     edge_idx = h_idx * (node_num-1) + i_idx
                # else:
                #     edge_idx = h_idx * (node_num-1) + i_idx -1

                hoi_ids = metadata.obj_hoi_index[metadata.coco_pytorch_to_coco[labels[i_idx]]]
                for hoi_idx in range(hoi_ids[0]-1, hoi_ids[1]):
                    act_idx = metadata.hoi_to_action[hoi_idx]
                    if act_prob[edge_idx][act_idx] < score_thresh:
                        continue
                    det_hoi_num += 1
                    x1,y1,x2,y2 = bboxs[h_idx]
                    x1_,y1_,x2_,y2_ = bboxs[i_idx]
                    if color_mark[h_idx] is None:
                        color_mark[h_idx] = (255, 0, 0)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_mark[h_idx], line_thickness)
                    if color_mark[i_idx] is None:
                        r_color = random.choice(np.arange(256))
                        g_color = random.choice(np.arange(256))
                        b_color = random.choice(np.arange(256))
                        color_mark[i_idx] = (int(r_color), int(g_color), int(b_color))
                        cv2.rectangle(img, (int(x1_), int(y1_)), (int(x2_), int(y2_)), color_mark[i_idx], line_thickness)
                    if show_line:
                        im_w,im_h = img.shape[:2][::-1]
                        c0 = int(0.5*x1)+int(0.5*x2)
                        r0 = int(0.5*y1)+int(0.5*y2)
                        c1 = int(0.5*x1_)+int(0.5*x2_)
                        r1 = int(0.5*y1_)+int(0.5*y2_)
                        c0 = max(0,min(c0,im_w-1))
                        c1 = max(0,min(c1,im_w-1))
                        r0 = max(0,min(r0,im_h-1))
                        r1 = max(0,min(r1,im_h-1))
                        cv2.line(img, (c0, r0), (c1, r1), color_mark[i_idx], line_thickness)
                    if show_pose and not pose_mark[h_idx] and pose is not None:
                        # import ipdb; ipdb.set_trace()
                        im_w,im_h = img.shape[:2][::-1]
                        keypoints = pose[h_idx]
                        for k, v in POSE_LINKED_LIST.items():
                            s_x, s_y = int(keypoints[k][0]), int(keypoints[k][1])
                            if s_x > im_w or s_y > im_h:
                                continue
                            cv2.circle(img, (s_x, s_y), 3, (0, 255, 0), -1)
                            if v is None:
                                continue
                            for i in v:
                                d_x, d_y = int(keypoints[i][0]), int(keypoints[i][1])
                                if d_x > im_w or d_y > im_h:
                                    continue        
                                cv2.line(img, (s_x, s_y), (d_x, d_y), (0, 255, 0), 2)                        
                        pose_mark[h_idx] = 1
                        
                    text = ' ' + metadata.hico_action_classes[act_idx] + ' ' + \
                                 metadata.coco_classes_pytorch[labels[i_idx]] + ' ' +\
                                 str(round(act_prob[edge_idx][act_idx], 3)) + \
                                 f'({round(scores[h_idx] * scores[i_idx] * act_prob[edge_idx][act_idx], 3)})' + ' '
                    retval, baseLine = cv2.getTextSize(text, fontFace=fontFace, fontScale=fontScale, thickness=font_thickness)
                    base_x = int(x1); base_y = int(y2)
                    topLeft = (base_x, base_y - (retval[1]+8) * (h_mark[h_idx]+1))
                    bottomRight = (topLeft[0] + retval[0], topLeft[1] + retval[1])
                    # import ipdb; ipdb.set_trace()
                    cv2.rectangle(img, (topLeft[0], topLeft[1]-3), (bottomRight[0], bottomRight[1]+3), thickness=-1, color=color_mark[i_idx])
                    cv2.putText(img, text, (topLeft[0], bottomRight[1]), fontScale=fontScale, fontFace=fontFace, thickness=font_thickness, color=(0,0,0))
                    h_mark[h_idx] += 1
        text = f'  det_obj: {node_num}({human_num})  det_hoi: {det_hoi_num}  ' 
        retval, baseLine = cv2.getTextSize(text, fontFace=fontFace, fontScale=fontScale, thickness=font_thickness)
        cv2.rectangle(img, (0, 0), (img.shape[:2][::-1][0], retval[1]+6), thickness=-1, color=(51, 255, 238))
        cv2.putText(img, text, (0, retval[1]+3), fontScale=fontScale, fontFace=fontFace, thickness=font_thickness, color=(0,0,0))
        return img
    except Exception as e:
        print("Error:", e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass


def vis_img_frcnn(img, bboxs, labels, scores=None, pose=None, score_thresh=0.8):
    if len(bboxs) == 0:
        return img    

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if scores is not None:
        keep = np.where(scores > score_thresh)[0]
        bboxs = bboxs[keep]
        labels = labels[keep]
        scores = scores[keep] 

    line_width = 1
    color = (120,0,0)
    # build the Font object
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=15)
    for idx, (bbox, label) in enumerate(zip(bboxs, labels)):
        Drawer = ImageDraw.Draw(img)
        Drawer.rectangle(list(bbox), outline=(120,0,0))
        text = metadata.coco_classes_pytorch[label]
        if scores is not None:
            text = text + " " + '{:.3f}'.format(scores[idx])
        h, w = font.getsize(text)
        Drawer.rectangle(xy=(bbox[0], bbox[1], bbox[0]+h+1, bbox[1]+w+1), fill=color, outline=None)
        Drawer.text(xy=(bbox[0], bbox[1]), text=text, font=font, fill=None)
    if pose is not None:
        for keypoints in pose:
            im_w,im_h = img.size
            for k, v in POSE_LINKED_LIST.items():
                s_x, s_y = int(keypoints[k][0]), int(keypoints[k][1])
                if s_x > im_w or s_y > im_h:
                    continue
                if v is None:
                    continue
                for i in v:
                    d_x, d_y = int(keypoints[i][0]), int(keypoints[i][1])
                    if d_x > im_w or d_y > im_h:
                        continue        
                    Drawer.line((s_x, s_y, d_x, d_y), fill=128, width=5)             
                
    return img