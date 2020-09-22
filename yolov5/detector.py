import cv2
import torch.backends.cudnn as cudnn

from utils.datasets import *
from utils.utils import *


def load_model(path):
    # Initialize
    device = torch_utils.select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(path, map_location=device)['model'].float()  # load to FP32s
    model.fuse()
    model.to(device).eval()
    
    if half:
        model.half()  # to FP16
    return model


def detect(img_path, model, device='0', img_size=640, iou_thres=0.5, conf_thres=0.4, agnostic_nms=True):
    img0 = cv2.imread(img_path)
    
    assert img0 is not None
    assert model is not None
    
    bbs = []
    with torch.no_grad():
        img0_shape = img0.shape
        imgsz = check_img_size(img_size, s=model.model[-1].stride.max())  # check img_size

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        device = torch_utils.select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        ### CHANGES
        # Padded resize
        img = letterbox(img0, new_shape=img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        ### CHANGES

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, agnostic=agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(img0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det) == 1:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0_shape).round()
                for *xyxy, conf, cls in det:
                    # normalized xywh
                    x_center, y_center, w, h = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                    x = x_center - w/2
                    y = y_center - h/2
                    img_h, img_w = img0_shape[:2]
                    x, y, w, h = int(x*img_w), int(y*img_h), int(w*img_w), int(h*img_h)
                    bbs.append((x, y, x+w, y+h))
    return bbs
