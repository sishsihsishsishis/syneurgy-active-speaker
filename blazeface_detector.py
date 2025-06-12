import torch
import numpy as np
import cv2
from blazeface_pytorch import BlazeFace

class BlazeFaceDetector:
    """
    A fast face detector using BlazeFace that follows the same interface as S3FD.
    """
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize BlazeFace detector
        self.detector = BlazeFace().to(device)
        self.detector.load_weights("blazeface.pth")
        self.detector.load_anchors("anchors.npy")
        self.detector.min_score_thresh = 0.75
        self.detector.min_suppression_threshold = 0.3
        
    def detect_faces(self, img, conf_th=0.9, scales=[1.0]):
        """
        Detect faces in the input image.
        
        Args:
            img: Input image (numpy array in RGB format)
            conf_th: Confidence threshold
            scales: List of scales to run detection at
        
        Returns:
            List of face bounding boxes in format [x1, y1, x2, y2, confidence]
        """
        height, width = img.shape[:2]
        result_boxes = []
        
        # Run detection at different scales
        for scale in scales:
            if scale != 1.0:
                resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
            else:
                resized_img = img
            
            # Convert to tensor format
            # BlazeFace expects 128x128 images
            resized_tensor = cv2.resize(resized_img, (128, 128))
            resized_tensor = np.transpose(resized_tensor, (2, 0, 1))
            resized_tensor = np.expand_dims(resized_tensor, axis=0)
            resized_tensor = torch.tensor(resized_tensor, dtype=torch.float32)
            resized_tensor = resized_tensor.to(self.device)
            
            # Normalize tensor
            resized_tensor = resized_tensor / 255.0
            
            # Run detection
            with torch.no_grad():
                detections = self.detector.predict_on_batch(resized_tensor)
            
            if detections[0].shape[0] == 0:
                continue
                
            # Convert to original image coordinates
            scale_h, scale_w = height / 128, width / 128
            
            for i in range(detections[0].shape[0]):
                ymin, xmin, ymax, xmax = detections[0][i, :4]
                score = detections[1][i]
                
                if score < conf_th:
                    continue
                
                # Adjust to current scale
                xmin = int(xmin * scale_w / scale)
                ymin = int(ymin * scale_h / scale)
                xmax = int(xmax * scale_w / scale)
                ymax = int(ymax * scale_h / scale)
                
                # Ensure box is within image boundaries
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width - 1, xmax)
                ymax = min(height - 1, ymax)
                
                result_boxes.append([xmin, ymin, xmax, ymax, float(score)])
        
        # Non-maximum suppression to remove overlapping boxes
        if len(result_boxes) > 0:
            result_boxes = self._nms(np.array(result_boxes), 0.3)
            
        return result_boxes
    
    def _nms(self, dets, thresh):
        """
        Non-maximum suppression to filter overlapping detections.
        """
        if len(dets) == 0:
            return []
            
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return dets[keep]