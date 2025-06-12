import numpy as np
import torch
from retinaface import RetinaFace
import cv2

class RetinaFaceDetector:
    """
    A faster face detector using RetinaFace that follows the same interface as S3FD.
    """
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize RetinaFace detector
        self.detector = RetinaFace(device=device)
        self.min_face_size = 20  # Minimum face size to detect
    
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
        
        # Run detection at different scales for better performance
        for scale in scales:
            if scale != 1.0:
                # Resize image for detection at this scale
                resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
            else:
                resized_img = img
                
            # Detect faces using RetinaFace
            faces = self.detector.detect(resized_img, threshold=conf_th)
            
            if faces is not None:
                for box, landmarks, score in faces:
                    if score < conf_th:
                        continue
                        
                    # Scale back to original image size
                    if scale != 1.0:
                        box = box / scale
                    
                    # Ensure box is within image boundaries
                    x1 = max(0, int(box[0]))
                    y1 = max(0, int(box[1]))
                    x2 = min(width - 1, int(box[2]))
                    y2 = min(height - 1, int(box[3]))
                    
                    # Skip small faces
                    if (x2 - x1 < self.min_face_size) or (y2 - y1 < self.min_face_size):
                        continue
                    
                    # Add to results
                    result_boxes.append([x1, y1, x2, y2, score])
        
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