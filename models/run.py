
import torch
import torch.nn as nn
import torchvision

import h5py
import numpy as np
from models import object_detector, vs_gats, pmn
from utils import metadata, vis_tool

OBJECT_DETECTOR_CP = './checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
POSE_DETECTOR_CP = './checkpoints/keypointrcnn_resnet50_fpn_coco-9f466800.pth'
VSGATS_CP = './checkpoints/vs_gats_checkpoint.pth'
PMN_CP = './checkpoints/pmn_checkpoint.pth'
WORD2VEC_DATA = './checkpoints/hico_word2vec.hdf5'

# './checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'

class HOI_DET_ONLINE_MODEL(nn.Module):
    def __init__(self, obj_det_only=False, use_pmn=False):
        super(HOI_DET_ONLINE_MODEL, self).__init__()

        self.obj_det_only = obj_det_only
        self.use_pmn = use_pmn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.word2vec = h5py.File(WORD2VEC_DATA, 'r')
        self.obj_pose_detector = self._load_obj_pose_detector(pretrained_obj=OBJECT_DETECTOR_CP, \
                                    pretrained_pose=POSE_DETECTOR_CP).to(self.device).eval()
        if not obj_det_only:
            self.hoi_detector_vsgats = self._load_hoi_detector_vsgats(pretrained=VSGATS_CP).to(self.device).eval()
            if use_pmn:
                self.hoi_detector_pmn = self._load_hoi_detector_pmn(pretrained=PMN_CP).to(self.device).eval()
        

    def forward(self, ori_img, action_threshold=0.5, show_line=True, show_pose=False):
        img = ori_img[:,:,::-1].copy()
        img_tensor = torchvision.transforms.functional.to_tensor(img).to(self.device)
        obj_pose_outputs, backbone_feats, img_sizes = self.obj_pose_detector([img_tensor])
        obj_boxes, obj_labels, obj_scores, pose = obj_pose_outputs[0]['boxes'], obj_pose_outputs[0]['labels'], \
                                                  obj_pose_outputs[0]['scores'], obj_pose_outputs[0]['keypoints']
        if self.obj_det_only:
            out_img = vis_tool.vis_img_frcnn(img, obj_boxes.cpu().detach().numpy(), obj_labels.cpu().detach().numpy(), pose=pose.cpu().detach().cpu().numpy(), score_thresh=0.8)
        else:
            # import ipdb; ipdb.set_trace()
            if not sum(obj_labels==1) or obj_labels.size(0) == 1:
                return ori_img
            obj_boxes_feats = self.obj_pose_detector.roi_heads.box_roi_pool(backbone_feats, [obj_boxes], img_sizes)
            obj_boxes_feats = self.obj_pose_detector.roi_heads.box_head(obj_boxes_feats)
            
            obj_boxes, obj_labels, obj_scores, obj_boxes_feats, pose = self._reconstruct_boxes(obj_boxes, obj_labels, \
                                                                                        obj_scores, obj_boxes_feats, pose, h_threshold=0.8)
            o2v_feats = torch.from_numpy(self._get_word2vec(obj_labels)).to(self.device)
            sp_feats = torch.from_numpy(self._calculate_spatial_feats(obj_boxes, img.shape[:2][::-1])).to(self.device)
            act_logits, _, _ = self.hoi_detector_vsgats([obj_boxes.shape[0]], obj_boxes_feats.float(), sp_feats.float(), o2v_feats.float(), [obj_labels])
            # NOTE! Need to fix the bug in pose detector 
            if self.use_pmn:
                ap_feats, rp_feats = self._calculate_pose_feats(obj_boxes, pose[:,:,:2], img.shape[:2][::-1], iter=True)
                ap_feats, rp_feats = torch.from_numpy(ap_feats).to(self.device).float(), torch.from_numpy(rp_feats).to(self.device).float()
                act_logits2 = self.hoi_detector_pmn(ap_feats, rp_feats)
                act_logits += act_logits2

            act_probs = nn.Sigmoid()(act_logits).cpu().detach().numpy()
            
            out_img = vis_tool.vis_img(img, obj_boxes, obj_labels, obj_scores, pose[:,:,:2], act_prob=act_probs, score_thresh=action_threshold, show_line=show_line, show_pose=show_pose)

        return np.array(out_img)[:,:,::-1]

    def _load_obj_pose_detector(self, pretrained_obj=None, pretrained_pose=None):
        obj_detector = object_detector.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained_obj, \
                                                                         rpn_post_nms_top_n_test=200, \
                                                                         box_batch_size_per_image=128, \
                                                                         box_score_thresh=0.4, box_nms_thresh=0.3)
        pose_detector = object_detector.detection.keypointrcnn_resnet50_fpn(pretrained=pretrained_pose, \
                                                                            rpn_post_nms_top_n_test=200, \
                                                                            box_batch_size_per_image=128, \
                                                                            box_score_thresh=0.3, box_nms_thresh=0.3)
        obj_detector.roi_heads.keypoint_roi_pool = pose_detector.roi_heads.keypoint_roi_pool
        obj_detector.roi_heads.keypoint_head = pose_detector.roi_heads.keypoint_head
        obj_detector.roi_heads.keypoint_predictor = pose_detector.roi_heads.keypoint_predictor

        return obj_detector

    def _load_hoi_detector_vsgats(self, pretrained=None):
        checkpoint = torch.load(pretrained)
        model = vs_gats.AGRNN(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], \
                                      bn=checkpoint['bn'], dropout=checkpoint['dropout'], \
                                      multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], \
                                      diff_edge=checkpoint['diff_edge'])
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def _load_hoi_detector_pmn(self, pretrained=None):
        pg_checkpoint = torch.load(pretrained)
        model = pmn.PGception(action_num=pg_checkpoint['a_n'], layers=1, classifier_mod=pg_checkpoint['classifier_mod'], o_c_l=pg_checkpoint['o_c_l'], \
                              last_h_c=pg_checkpoint['last_h_c'], bias=pg_checkpoint['bias'], drop=pg_checkpoint['dropout'], bn=pg_checkpoint['bn'], \
                              agg_first=pg_checkpoint['agg_first'], attn=pg_checkpoint['attn'], b_l=pg_checkpoint['b_l'])
        model.load_state_dict(pg_checkpoint['state_dict'])

        return model

    def _reconstruct_boxes(self, boxes, labels, scores, feats, pose, h_threshold=0.8):
        '''
            To omit the background boxes and arrange the boxes from human to object
        '''
        clean_list = [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
        n_boxes, n_labels, n_scores, n_feats = torch.empty((0,4)), [], [], torch.empty((0, 1024))
        
        for i, label in enumerate(labels):
            if label in clean_list:
                continue
            n_boxes = torch.cat((n_boxes, boxes[i][None].cpu().detach()))
            n_labels.append(labels[i].item())
            n_scores.append(scores[i].item())
            n_feats = torch.cat((n_feats, feats[i][None].cpu().detach()))

        l_idx = 0
        h_omit_list = [] 
        for r_idx in range(n_boxes.size(0)):
            if n_labels[r_idx] != 1:
                continue
            if l_idx != r_idx:
                n_boxes[[l_idx, r_idx], :] = n_boxes[[r_idx, l_idx], :]
                n_labels[l_idx], n_labels[r_idx] = n_labels[r_idx], n_labels[l_idx]
                n_scores[l_idx], n_scores[r_idx] = n_scores[r_idx], n_scores[l_idx]
                n_feats[[l_idx, r_idx], :] = n_feats[[r_idx, l_idx], :]
                if n_scores[l_idx] < h_threshold:
                    h_omit_list.append(l_idx)
            l_idx += 1

        n_boxes = n_boxes.numpy()
        n_labels = np.array(n_labels)
        n_scores = np.array(n_scores)
        n_feats = n_feats.numpy()
        n_pose = pose.cpu().detach().numpy()
        if len(h_omit_list) and len(h_omit_list) != sum(n_labels==1):
            n_boxes = np.delete(n_boxes, h_omit_list, axis=0)
            n_labels = np.delete(n_labels, h_omit_list, axis=0)
            n_scores = np.delete(n_scores, h_omit_list, axis=0)
            n_feats = np.delete(n_feats, h_omit_list, axis=0)
            n_pose = np.delete(n_pose, h_omit_list, axis=0)
        
        return n_boxes, n_labels, n_scores, torch.from_numpy(n_feats).to(self.device), n_pose
            
    def _get_word2vec(self, node_ids):
        word2vec = np.empty((0,300))
        for node_id in node_ids:
            vec = self.word2vec[metadata.coco_classes_pytorch[node_id]]
            word2vec = np.vstack((word2vec, vec))
        return word2vec

    def _calculate_spatial_feats(self, det_boxes, im_wh):
        spatial_feats = []
        for i in range(det_boxes.shape[0]):
            for j in range(det_boxes.shape[0]):
                if j == i:
                    continue
                else:
                    single_feat = []
                    box1_wrt_img = self._box_with_respect_to_img(det_boxes[i], im_wh)
                    box2_wrt_img = self._box_with_respect_to_img(det_boxes[j], im_wh)
                    box1_wrt_box2 = self._box1_with_respect_to_box2(det_boxes[i], det_boxes[j])
                    offset = self._center_offset(det_boxes[i], det_boxes[j], im_wh)
                    single_feat = single_feat + box1_wrt_img + box2_wrt_img + box1_wrt_box2 + offset.tolist()
                    spatial_feats.append(single_feat)
        spatial_feats = np.array(spatial_feats)
        return spatial_feats

    def _calculate_pose_feats(self, det_boxes, keypoints, im_wh, iter=True):
        pose_to_human = []
        pose_to_obj_offset = []
        for h_idx in range(keypoints.shape[0]):
            for o_idx in range(det_boxes.shape[0]):
                if iter:
                    if o_idx <= h_idx:  continue
                    else:
                        pose_to_human.append(self._cal_pose_to_box(keypoints[h_idx], det_boxes[h_idx]))
                        pose_to_obj_offset.append(self._cal_pose_to_box_offset(keypoints[h_idx], det_boxes[o_idx], im_wh))
                else:
                    if o_idx == h_idx:  continue
                    else:
                        pose_to_human.append(self._cal_pose_to_box(keypoints[h_idx], det_boxes[h_idx]))
                        pose_to_obj_offset.append(self._cal_pose_to_box_offset(keypoints[h_idx], det_boxes[o_idx], im_wh))                        
        pose_to_human = np.array(pose_to_human)
        pose_to_obj_offset = np.array(pose_to_obj_offset)
        return pose_to_human, pose_to_obj_offset

    def _center_offset(self, box1, box2, im_wh):
        c1 = [(box1[2]+box1[0])/2, (box1[3]+box1[1])/2]
        c2 = [(box2[2]+box2[0])/2, (box2[3]+box2[1])/2]
        offset = np.array(c1)-np.array(c2)/np.array(im_wh)
        return offset

    def _box_with_respect_to_img(self, box, im_wh):
        '''
            To get [x1/W, y1/H, x2/W, y2/H, A_box/A_img]
        '''
        feats = [box[0]/(im_wh[0]+ 1e-6), box[1]/(im_wh[1]+ 1e-6), box[2]/(im_wh[0]+ 1e-6), box[3]/(im_wh[1]+ 1e-6)]
        box_area = (box[2]-box[0])*(box[3]-box[1])
        img_area = im_wh[0]*im_wh[1]
        feats +=[ box_area/(img_area+ 1e-6) ]
        return feats

    def _box1_with_respect_to_box2(self, box1, box2):
        feats = [ (box1[0]-box2[0])/(box2[2]-box2[0]+1e-6),
                (box1[1]-box2[1])/(box2[3]-box2[1]+ 1e-6),
                np.log((box1[2]-box1[0])/(box2[2]-box2[0]+ 1e-6)),
                np.log((box1[3]-box1[1])/(box2[3]-box2[1]+ 1e-6))   
                ]
        return feats

    def _cal_pose_to_box(self, keypoint, box):
        feat = 2 * keypoint / np.array([box[0]+box[2], box[1]+box[3]])
        return feat

    def _cal_pose_to_box_offset(self, keypoint, box, im_wh):
        feat = (keypoint - (np.array([box[0]+box[2], box[1]+box[3]])/2)) / np.array(im_wh)
        return feat