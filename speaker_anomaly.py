import csv
import pandas as pd
import numpy as np
import torch

class SpeakerIDAnomalyDetector:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.total_bbox_area = np.sum((self.data['x2'] - self.data['x1']) * (self.data['y2'] - self.data['y1']))
        
    def detect_anomalies(self, threshold=0.15):
        # Convert to torch tensors for faster calculation
        data_tensor = torch.tensor(self.data[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float32)
        
        # Calculate bbox areas using tensor operations
        bbox_areas = (data_tensor[:, 2] - data_tensor[:, 0]) * (data_tensor[:, 3] - data_tensor[:, 1])
        total_bbox_area = bbox_areas.sum().item()
        
        # Group by speakerID
        speaker_ids = self.data['speakerID'].unique()
        anomalies = []
        
        for speaker in speaker_ids:
            # Get indices for this speaker
            speaker_mask = (self.data['speakerID'] == speaker).values
            speaker_areas = bbox_areas[speaker_mask]
            
            # Calculate total area and duration
            speaker_total_area = speaker_areas.sum().item()
            speaker_duration = len(speaker_areas)
            
            if speaker_total_area < total_bbox_area * threshold:
                anomalies.append({
                    'speakerID': speaker,
                    'total_area': speaker_total_area,
                    'duration': speaker_duration,
                    'percentage_area': speaker_total_area / total_bbox_area
                })
        
        return anomalies
    
    def print_anomalies(self, anomalies):
        print("Detected anomalies:")
        for anomaly in anomalies:
            print(f"SpeakerID: {anomaly['speakerID']}, Total Area: {anomaly['total_area']}, "
                  f"Duration: {anomaly['duration']}, Percentage Area: {anomaly['percentage_area']:.2%}")

    def correct_anomalies(self, anomalies):
        corrected_data = self.data.copy()
        for anomaly in anomalies:
            speaker_id = anomaly['speakerID']
            anomaly_indices = corrected_data[corrected_data['speakerID'] == speaker_id].index
            
            # Find the closest speaker by timestamp and bounding box overlap
            for idx in anomaly_indices:
                timestamp = corrected_data.at[idx, 'timestamp']
                bbox = corrected_data.loc[idx, ['x1', 'y1', 'x2', 'y2']].values

                # Calculate the distance to other bounding boxes
                distances = []
                for other_idx, row in corrected_data[corrected_data['speakerID'] != speaker_id].iterrows():
                    other_bbox = row[['x1', 'y1', 'x2', 'y2']].values
                    iou = self.bb_intersection_over_union(bbox, other_bbox)
                    distances.append((iou, row['speakerID']))

                # Find the best match
                if distances:
                    best_match = max(distances, key=lambda x: x[0])
                    if best_match[0] > 0.5:  # IOU threshold
                        corrected_data.at[idx, 'speakerID'] = best_match[1]

        return corrected_data

    def save_corrected_data(self, corrected_data, output_path='exports/corrected_detection.csv'):
        corrected_data.to_csv(output_path, index=False)
        print(f"Corrected data saved to {output_path}")

    def bb_intersection_over_union(self, boxes_a, boxes_b):
        """Calculate IoU between all pairs of boxes in boxes_a and boxes_b"""
        # Convert to torch tensors if they aren't already
        if not isinstance(boxes_a, torch.Tensor):
            boxes_a = torch.tensor(boxes_a, dtype=torch.float32)
        if not isinstance(boxes_b, torch.Tensor):
            boxes_b = torch.tensor(boxes_b, dtype=torch.float32)
        
        # Expand dimensions to calculate IoU between all pairs
        boxes_a = boxes_a.unsqueeze(1)  # Shape: (N, 1, 4)
        boxes_b = boxes_b.unsqueeze(0)  # Shape: (1, M, 4)
        
        # Calculate intersection coordinates
        x_left = torch.max(boxes_a[:, :, 0], boxes_b[:, :, 0])
        y_top = torch.max(boxes_a[:, :, 1], boxes_b[:, :, 1])
        x_right = torch.min(boxes_a[:, :, 2], boxes_b[:, :, 2])
        y_bottom = torch.min(boxes_a[:, :, 3], boxes_b[:, :, 3])
        
        # Calculate area of intersection
        intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
        
        # Calculate area of boxes
        area_a = (boxes_a[:, :, 2] - boxes_a[:, :, 0]) * (boxes_a[:, :, 3] - boxes_a[:, :, 1])
        area_b = (boxes_b[:, :, 2] - boxes_b[:, :, 0]) * (boxes_b[:, :, 3] - boxes_b[:, :, 1])
        
        # Calculate IoU
        union_area = area_a + area_b - intersection_area
        iou = intersection_area / union_area
        
        return iou

# Example usage:
detector = SpeakerIDAnomalyDetector('exports/raw_detection.csv')
anomalies = detector.detect_anomalies()
detector.print_anomalies(anomalies)
corrected_data = detector.correct_anomalies(anomalies)
detector.save_corrected_data(corrected_data)
