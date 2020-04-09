# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker


from . import discriminative_features as df
from math import ceil
from math import floor
import cv2


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 
        #self.score_size = 17
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        #print ('score size is',self.score_size)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        #print(bbox)
        #print (img.shape)
        

        # rows,cols,chan=img.shape
        # origin=np.array([cols/2,rows/2])
        # new_bbox=np.zeros(4);
        # new_bbox[2]=bbox[2]
        # new_bbox[3]=bbox[3]
        # new_bbox[0]=origin[0]-bbox[0]
        # new_bbox[1]=origin[1]-bbox[1]

        # y1 = bbox[1]
        # y2 = bbox[3] +y1
        # x1 = bbox[0]
        # x2 = bbox[2] +x1
        # #cropped=img[floor(min_x):floor(min_x+bbox[2]),floor(min_y):floor(min_y+bbox[3])]
        # cropped = img[floor(y1):floor(y2), floor(x1):floor(x2)]
        # print (cropped.shape)
        # cv2.imshow("Cropped",cropped)
        # df.__init__(img,bbox);
        #import pdb; pdb.set_trace()
        
        self.bbox=bbox;
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        self.count = 0

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average,self.bbox)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        #import pdb; pdb.set_trace()
        #cv2.imshow("IMage",img)
        #print("image size: ",img.shape)
        self.count += 1
        #       print(self.count)
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        #import pdb;pdb.set_trace()
        #print(self.bbox)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average,self.bbox)
        #import pdb;pdb.set_trace()
        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        #import pdb; pdb.set_trace()
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        #import pdb; pdb.set_trace()
        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox_new = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        cc,rr,ww,hh = bbox_new[0], bbox_new[1], bbox_new[2], bbox_new[3]
        #import pdb;pdb.set_trace()
        from math import floor
        track_window = (floor(bbox_new[0]), floor(bbox_new[1]), floor(bbox_new[2]), floor(bbox_new[3]))
        '''
        cc = floor(bbox_new[0])
        rr = floor(bbox_new[1])
        ww = floor(bbox_new[2])
        hh = floor(bbox_new[3])
        '''
        #import pdb; pdb.set_trace()
        #roi = img[rr:rr+hh, cc:cc+ww]
        #roi = img[ceil(rr):ceil(rr+hh), ceil(cc):ceil(cc+ww)]
        '''
        hsv_roi =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        #import pdb; pdb.set_trace()
        track_window = (floor(bbox_new[0]), floor(bbox_new[1]), floor(bbox_new[2]), floor(bbox_new[3]))
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        #print('track_window')
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(img, (x,y), (x+w,y+h), 255,2)
        #cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        return_value = cv2.imwrite(chr(k)+".jpg",img2)
        print ("bbox",bbox)
        print(img.shape)
        self.bbox=bbox;
        #cv.imshow("ROI", img[bbox[0]:bbox]);
        #import pdb; pdb.set_trace()
        '''
        self.bbox=np.array(track_window);
        return {
                'bbox': np.array(track_window),
                'best_score': best_score
               }
