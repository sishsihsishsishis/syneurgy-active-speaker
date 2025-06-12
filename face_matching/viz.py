import cv2
import csv
import os
from tqdm import tqdm

def visualize_results(video_path, csv_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Read the CSV file and store the data
    detections = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            detections.append(row)

    # Process the video frame by frame
    frame_idx = 0
    current_timestamp = 0.0
    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current timestamp in seconds
        timestamp = frame_idx / fps

        # Draw bounding boxes for the current timestamp
        for detection in detections:
            det_timestamp = float(detection['timestamp'])
            if abs(det_timestamp - timestamp) < 1 / fps:
                user_id = detection['User_ID']
                x1, y1, x2, y2 = int(detection['X1']), int(detection['Y1']), int(detection['X2']), int(detection['Y2'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, user_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame (optional)
        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_idx += 1

    # Release the video objects
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'data/video/gitlab2.mp4'  # Path to your input video
    csv_path = 'exports/active_speakers.csv'  # Path to the CSV file with detections
    output_video_path = 'exports/annotated_video.mp4'  # Path to save the output video

    # Create results directory if it doesn't exist
    if not os.path.exists('exports'):
        os.makedirs('exports')

    visualize_results(video_path, csv_path, output_video_path)
    print(f"Annotated video saved to {output_video_path}")
