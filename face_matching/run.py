import sys
import os
import glob
import traceback
import shutil
import csv
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cdist
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_fresh_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def save_results_to_csv(user_id, bbox, timestamp, output_csv):
    if user_id != "Unknown" and bbox is not None:
        bbox_int = [int(coord) for coord in bbox]
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([user_id, timestamp, *bbox_int])

def load_reference_encodings(reference_image_folder):
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    reference_encodings = []
    reference_user_ids = []

    for image_path in glob.glob(os.path.join(reference_image_folder, '*.jpg')):
        user_id = os.path.basename(image_path).split('.')[0]
        img = Image.open(image_path).convert('RGB')
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            if isinstance(img_cropped, list):
                img_cropped = torch.stack(img_cropped)  # Stack list of tensors to a single tensor
            if img_cropped.ndim == 3:
                img_cropped = img_cropped.unsqueeze(0)  # Convert to batch of size 1 if not already batched
            with torch.no_grad():
                encoding = resnet(img_cropped.to(device)).cpu().numpy()
            reference_encodings.append(encoding)
            reference_user_ids.append(user_id)
    
    return reference_encodings, reference_user_ids

def match_faces(face_encodings, reference_encodings, reference_user_ids, threshold=0.8):
    if len(face_encodings) == 0:
        return []
    distances = cdist(face_encodings, np.vstack(reference_encodings), metric='euclidean')
    matched_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    matched_user_ids = [reference_user_ids[idx] if min_distances[i] < threshold else 'Unknown'
                        for i, idx in enumerate(matched_indices)]
    return matched_user_ids

def process_video(video_path, reference_encodings, reference_user_ids, output_csv, frame_interval=2):
    create_fresh_directory(os.path.dirname(output_csv))

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['User_ID', 'timestamp', 'X1', 'Y1', 'X2', 'Y2'])
        
    cap = cv2.VideoCapture(video_path)
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = int(frame_interval * fps)

    current_time = 0
    for frame_idx in tqdm(range(0, frame_count, skip_frames), desc='Processing video'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]
        try:
            img_cropped_list, _ = mtcnn(imgs, return_prob=True)
        except Exception as e:
            print(f"Error in MTCNN face detection: {e}")
            img_cropped_list = None
        
        if img_cropped_list is not None and any([img is not None for img in img_cropped_list]):
            face_encodings = []
            for img_cropped in img_cropped_list:
                if img_cropped is not None:
                    if img_cropped.ndim == 3:
                        img_cropped = img_cropped.unsqueeze(0)  # Convert to batch of size 1 if not already batched
                    with torch.no_grad():
                        encoding = resnet(img_cropped.to(device)).cpu().numpy()
                    face_encodings.append(encoding)
            
            if face_encodings:
                face_encodings = np.vstack(face_encodings)
                matched_user_ids = match_faces(face_encodings, reference_encodings, reference_user_ids)

                for user_id, bbox in zip(matched_user_ids, mtcnn.detect(imgs)[0][0]):
                    save_results_to_csv(user_id, bbox, current_time, output_csv)
        
        current_time += frame_interval
    
    cap.release()

def main():
    video_folder = 'data/video/'
    reference_image_folder = 'data/users/'
    output_csv = 'exports/active_speakers.csv'

    try:
        video_path = glob.glob(os.path.join(video_folder, '*.mp4'))[0]
        print(f'video : {video_path}\n\n\n')
    except IndexError:
        print("No video file found in the media folder.")
        sys.exit(1)

    create_fresh_directory('results')

    try:
        reference_encodings, reference_user_ids = load_reference_encodings(reference_image_folder)
    except Exception as e:
        print(f"Error loading reference images: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        process_video(video_path, reference_encodings, reference_user_ids, output_csv)
    except Exception as e:
        print(f"Error processing video: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
