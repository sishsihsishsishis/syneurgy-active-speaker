import cv2
import pandas as pd
import traceback

# Load the CSV file
csv_path = 'exports/corrected_detection.csv'
video_path = 'data/demo.mp4'
output_video_path = 'exports/output_video_with_bounding_boxes.mp4'

try:
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Read and process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current timestamp in the video
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Filter the rows in the CSV file for the current timestamp
        relevant_rows = df[(df['timestamp'] <= current_time) & (df['timestamp'] > current_time - 1.0 / fps)]
        
        # Draw bounding boxes for the relevant rows
        for _, row in relevant_rows.iterrows():
            speaker_id = row['speakerID']
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put the speaker ID label
            cv2.putText(frame, speaker_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("An error occurred:")
    traceback.print_exc()
