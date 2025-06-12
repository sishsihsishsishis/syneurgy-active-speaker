import cv2
import pandas as pd

class VideoBoundingBoxVisualizer:
    def __init__(self, video_path, csv_path, output_path):
        self.video_path = video_path
        self.csv_path = csv_path
        self.output_path = output_path
        self.data = pd.read_csv(self.csv_path)
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            self.output_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            self.fps, 
            (self.frame_width, self.frame_height)
        )

    def draw_bounding_boxes(self):
        frame_index = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = frame_index / self.fps
            for _, row in self.data.iterrows():
                start_time = row['Start_time']
                end_time = row['End_time']
                if start_time <= current_time <= end_time:
                    bbox = (int(row['X1']), int(row['Y1']), int(row['X2']), int(row['Y2']))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    print(f"Timestamp: {current_time:.1f}s, BBox: {bbox}")

            self.out.write(frame)
            cv2.imshow('Video', frame)
            frame_index += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release_resources(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

# Usage
video_path = 'data/demo.mp4'
csv_path = 'exports/active_speakers.csv'
output_path = 'exports/no_trust_with_bboxes.mp4'
visualizer = VideoBoundingBoxVisualizer(video_path, csv_path, output_path)
visualizer.draw_bounding_boxes()
visualizer.release_resources()
