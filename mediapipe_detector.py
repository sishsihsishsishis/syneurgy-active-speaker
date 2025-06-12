import mediapipe as mp
import numpy as np
import cv2
import time

class MediaPipeFaceDetector:
    """
    A highly optimized face detector using MediaPipe that follows the same interface as S3FD.
    """
    def __init__(self, device='cpu'):
        # MediaPipe works efficiently on CPU
        self.mp_face_detection = mp.solutions.face_detection
        # Initialize with optimal settings for production use
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = full range model (better for various face sizes)
            min_detection_confidence=0.5  # Can be adjusted based on needs
        )
        
    def detect_faces(self, img, conf_th=0.5, scales=[0.5]):
        """
        Detect faces in the input image using MediaPipe.
        
        Args:
            img: Input image (numpy array in RGB format)
            conf_th: Confidence threshold
            scales: List of scales to run detection at
        
        Returns:
            List of face bounding boxes in format [x1, y1, x2, y2, confidence]
        """
        height, width = img.shape[:2]
        result_boxes = []
        
        # Process at efficient scale
        for scale in scales:
            if scale != 1.0:
                # Resize image for detection
                resized_height = int(height * scale)
                resized_width = int(width * scale)
                resized_img = cv2.resize(img, (resized_width, resized_height))
            else:
                resized_img = img
                resized_height = height
                resized_width = width
                
            # MediaPipe requires non-writeable images
            resized_img.flags.writeable = False
            
            # Detect faces
            results = self.detector.process(resized_img)
            
            # Make image writeable again
            resized_img.flags.writeable = True
            
            if results.detections:
                for detection in results.detections:
                    score = detection.score[0]
                    if score < conf_th:
                        continue
                        
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * resized_width)
                    y1 = int(bbox.ymin * resized_height)
                    w = int(bbox.width * resized_width)
                    h = int(bbox.height * resized_height)
                    x2 = x1 + w
                    y2 = y1 + h
                    
                    # Scale back to original image dimensions
                    if scale != 1.0:
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                    
                    # Ensure box is within image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width - 1, x2)
                    y2 = min(height - 1, y2)
                    
                    # Add to results with confidence score
                    result_boxes.append([x1, y1, x2, y2, float(score)])
        
        # Apply non-maximum suppression for overlapping boxes
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