import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import sys
# print(os.path.dirname(__file__))
# sys.path.append(os.path.dirname(__file__))
from .data.config import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm
from .utils.timer import Timer

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class Detector():
    def __init__(self, network="mobile0.25", trained_model='./weights/Resnet50_Final.pth',
                cpu=False, confidence_threshold=0.02, nms_threshold=0.4, detector_thresh=0.9):
        self.detector_thresh = detector_thresh
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        torch.set_grad_enabled(False)
        self.cfg = None
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        net = RetinaFace(cfg=self.cfg, phase = 'test')
        net = load_model(net, trained_model, cpu)
        net.eval()
        print('Finished loading model!')
        print(net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = net.to(self.device)

       

        # testing begin
    def detect(self, img_raw):
        _t = {'forward_pass': Timer(), 'misc': Timer()}
        img = np.float32(img_raw)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        _t['forward_pass'].tic()
        loc, conf, landms = self.net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]
        probs = dets[:, 4]
        # dets_show = np.concatenate((dets, landms), axis=1)
        dets = dets[:, : 4]
        detctions_threshed = np.where(probs > self.detector_thresh)
        probs = probs[detctions_threshed]
        dets = dets[detctions_threshed]
        landms = landms[detctions_threshed]
        _t['misc'].toc()
        return dets, probs, landms

if __name__ == '__main__':
    path = r"E:\work\e-concierge\Track DS 1.0\track_Person0027_01"
    detector = Detector(trained_model='weights\mobilenet0.25_Final.pth', cpu=True)
    for name in os.listdir(path):
        if not name.endswith('.png'): continue
        im = cv2.imread(os.path.join(path, name))
        dets, probs, landms = detector.detect(im)
        dets_where = np.where(probs > 0.9)
        print(dets[dets_where].astype(int), landms[dets_where].astype(int), probs[dets_where])
        
