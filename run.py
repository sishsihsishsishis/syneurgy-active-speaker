import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features, csv, traceback
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
#from model.faceDetector.s3fd import S3FD
from torch.cuda.amp import autocast, GradScaler

from ASD import ASD
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cdist
from PIL import Image
import concurrent
import threading

from torch.cuda.amp import autocast


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description = "Columbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="col",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="colDataPath",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="weight/pretrain_AVA_CVPR.model",   help='Path for the pretrained model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=5000,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columbia dataset')
parser.add_argument('--colSavePath',           type=str, default="/colDataPath",  help='Path for inputs, tmps and outputs')

args = parser.parse_args()


if args.evalCol == True:
	args.videoName = 'col'
	args.videoFolder = args.colSavePath
	args.savePath = os.path.join(args.videoFolder, args.videoName)
	args.videoPath = os.path.join(args.videoFolder, args.videoName + '.mp4')
	args.duration = 0
	if os.path.isfile(args.videoPath) == False:  # Download video
		link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
		cmd = "youtube-dl -f best -o %s '%s'"%(args.videoPath, link)
		output = subprocess.call(cmd, shell=True, stdout=None)
	if os.path.isdir(args.videoFolder + '/col_labels') == False: # Download label
		link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
		cmd = "gdown --id %s -O %s"%(link, args.videoFolder + '/col_labels.tar.gz')
		subprocess.call(cmd, shell=True, stdout=None)
		cmd = "tar -xzvf %s -C %s"%(args.videoFolder + '/col_labels.tar.gz', args.videoFolder)
		subprocess.call(cmd, shell=True, stdout=None)
		os.remove(args.videoFolder + '/col_labels.tar.gz')	
else:
	args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
	args.savePath = os.path.join(args.videoFolder, args.videoName)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_reference_encodings(reference_image_folder, batch_size=32):
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    image_paths = glob.glob(os.path.join(reference_image_folder, '*.jpg'))
    user_ids = [os.path.basename(path).split('.')[0] for path in image_paths]
    
    # Initialize lists to store results
    reference_encodings = []
    reference_user_ids = []
    
    # Process images in batches where possible
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_ids = user_ids[i:i+batch_size]
        batch_imgs = [Image.open(path).convert('RGB') for path in batch_paths]
        
        # Process each image (keeping original MTCNN interface)
        for img, user_id in zip(batch_imgs, batch_ids):
            img_cropped = mtcnn(img)
            if img_cropped is not None:
                if isinstance(img_cropped, list):
                    img_cropped = torch.stack(img_cropped)  # Stack list of tensors
                if img_cropped.ndim == 3:
                    img_cropped = img_cropped.unsqueeze(0)  # Add batch dimension
                
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

def visualization(tracks, scores, args, reference_encodings, reference_user_ids):
    """
    Process detection results and output to CSV, with optimized logging and processing.
    """
    print(f"Starting to process visualization for {len(tracks)} tracks...")
    start_time = time.time()
    
    # Ensure export directory exists and is fresh
    export_dir = 'exports'
    if os.path.exists(export_dir):
        rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    # Define the output CSV file path
    output_csv = os.path.join(export_dir, 'raw_detection.csv')

    # Prepare MTCNN and ResNet models for face recognition
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Get all frames in advance
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()

    # Calculate FPS from the video file
    video = cv2.VideoCapture(args.videoFilePath)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    # Process all tracks in batch
    speaker_id_mapping = {}  # To track face-to-speakerID mappings
    next_speaker_id = 0  # Counter for incremental speaker IDs
    
    # Collect all active speaker detections in memory first
    active_speaker_detections = []
    
    # Process all tracks
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            # Calculate average score for this frame
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
            s = np.mean(s)
            
            if s >= 0:  # Active speaker
                timestamp = frame / fps
                x1 = int(track['proc_track']['x'][fidx] - track['proc_track']['s'][fidx])
                y1 = int(track['proc_track']['y'][fidx] - track['proc_track']['s'][fidx])
                x2 = int(track['proc_track']['x'][fidx] + track['proc_track']['s'][fidx])
                y2 = int(track['proc_track']['y'][fidx] + track['proc_track']['s'][fidx])
                
                # Store detection for batch processing
                active_speaker_detections.append({
                    'track_idx': tidx,
                    'frame': frame,
                    'timestamp': timestamp,
                    'bbox': (x1, y1, x2, y2)
                })
    
    # Group detections by frames for batch processing
    detections_by_frame = {}
    for det in active_speaker_detections:
        frame = det['frame']
        if frame not in detections_by_frame:
            detections_by_frame[frame] = []
        detections_by_frame[frame].append(det)
    
    # Process faces in batches to optimize GPU usage
    print(f"Processing {len(active_speaker_detections)} detections across {len(detections_by_frame)} frames...")
    
    # Batch process faces by frame
    results = []
    frames_processed = 0
    
    # Process in batches of frames
    batch_size = 32  # Process 10 frames at a time
    frame_numbers = sorted(detections_by_frame.keys())
    
    for i in range(0, len(frame_numbers), batch_size):
        batch_frames = frame_numbers[i:i+batch_size]
        batch_detections = []
        
        # Collect all detections for this batch of frames
        for frame in batch_frames:
            batch_detections.extend(detections_by_frame[frame])
        
        # Process faces in this batch
        batch_results = []
        for det in batch_detections:
            image_path = flist[det['frame']]
            x1, y1, x2, y2 = det['bbox']
            
            # Load and crop face
            image = Image.open(image_path).convert('RGB')
            img_cropped = image.crop((x1, y1, x2, y2))
            img_cropped = mtcnn(img_cropped)
            
            if img_cropped is not None:
                if img_cropped.ndim == 3:
                    img_cropped = img_cropped.unsqueeze(0)
                
                # Get face encoding
                with torch.no_grad():
                    with autocast():
                        face_encoding = resnet(img_cropped.to(device)).cpu().numpy()
                
                # Match face (handle empty reference_encodings case)
                if len(reference_encodings) > 0:
                    try:
                        distances = cdist(face_encoding, np.vstack(reference_encodings), metric='euclidean')
                        matched_idx = np.argmin(distances, axis=1)[0]
                        min_distance = np.min(distances, axis=1)[0]
                        
                        # Use threshold to determine if match is valid
                        if min_distance < 0.8:  # Adjust threshold as needed
                            user_id = reference_user_ids[matched_idx]
                        else:
                            user_id = 'Unknown'
                    except:
                        user_id = 'Unknown'
                else:
                    user_id = 'Unknown'
                
                # Maintain consistent speaker IDs
                if user_id in speaker_id_mapping:
                    speaker_id = speaker_id_mapping[user_id]
                else:
                    speaker_id = f'speaker_{next_speaker_id}'
                    speaker_id_mapping[user_id] = speaker_id
                    next_speaker_id += 1
                
                # Add to results
                batch_results.append({
                    'speakerID': speaker_id,
                    'timestamp': round(det['timestamp'], 2),
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })
        
        # Add batch results to overall results
        results.extend(batch_results)
        frames_processed += len(batch_frames)
        
        # Periodic progress update (not too frequent)
        if frames_processed % 50 == 0:
            print(f"Processed {frames_processed}/{len(frame_numbers)} frames...")
    
    # Write final results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['speakerID', 'timestamp', 'x1', 'y1', 'x2', 'y2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort results by timestamp for cleaner output
        results.sort(key=lambda x: x['timestamp'])
        
        # Write all results
        for result in results:
            writer.writerow(result)
    
    # Final summary - minimal logging
    end_time = time.time()
    print(f"Visualization complete! Processed {len(results)} detections in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_csv}")


def scene_detect(args):
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(args.pyworkPath, 'scene.pckl')
    if not sceneList:
        sceneList = [(videoManager.get_base_timecode(), videoManager.get_current_timecode())]
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write(f'{args.videoFilePath} - scenes detected {len(sceneList)}\n')
    return sceneList

def inference_video(args, batch_size=16):
    """
    Detect faces in video frames using MediaPipe's optimized face detector.
    Processes frames in batches for improved performance.
    """
    # Import and initialize the detector
    from mediapipe_detector import MediaPipeFaceDetector
    DET = MediaPipeFaceDetector()
    
    # Get all frame paths
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    total_frames = len(flist)
    
    # Initialize empty detection lists for all frames
    dets = [[] for _ in range(total_frames)]
    
    # Start timing
    print(f"Starting face detection on {total_frames} frames with batch size {batch_size}...")
    start_time = time.time()
    
    # Process frames in batches
    for batch_start in range(0, total_frames, batch_size):
        batch_end = min(batch_start + batch_size, total_frames)
        batch_indices = list(range(batch_start, batch_end))
        batch_fnames = [flist[i] for i in batch_indices]
        
        # Process each frame in the batch
        for i, (fname, idx) in enumerate(zip(batch_fnames, batch_indices)):
            # Read and convert image
            image = cv2.imread(fname)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run face detection
            bboxes = DET.detect_faces(
                image_rgb, 
                conf_th=0.5, 
                scales=[args.facedetScale]
            )
            
            # Store detections
            for bbox in bboxes:
                dets[idx].append({
                    'frame': idx, 
                    'bbox': (bbox[:-1]).tolist(), 
                    'conf': bbox[-1]
                })
        
        # Print progress less frequently to avoid log spam
        if batch_end % 100 == 0 or batch_end == total_frames:
            elapsed = time.time() - start_time
            fps = batch_end / elapsed if elapsed > 0 else 0
            print(f"Processed {batch_end}/{total_frames} frames ({fps:.2f} fps)")
    
    # Save detections to pickle file
    save_path = os.path.join(args.pyworkPath, 'faces.pckl')
    with open(save_path, 'wb') as fil:
        pickle.dump(dets, fil)
    
    # Report final statistics
    total_time = time.time() - start_time
    avg_fps = total_frames / total_time if total_time > 0 else 0
    print(f"Face detection completed: {total_frames} frames in {total_time:.2f} seconds ({avg_fps:.2f} fps)")
    
    return dets

def bb_intersection_over_union(boxA, boxB, evalCol=False):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def track_shot(args, sceneFaces):
    iouThres = 0.5
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if not track:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]), numpy.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks

def crop_video(args, track, cropFile):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = args.cropScale
        bs = dets['s'][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        frame = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi
        mx = dets['x'][fidx] + bsi
        face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + '.wav'
    audioStart = (track['frame'][0]) / 25
    audioEnd = (track['frame'][-1] + 1) / 25
    vOut.release()
    command = f"ffmpeg -y -i {args.audioFilePath} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads {args.nDataLoaderThread} -ss {audioStart:.3f} -to {audioEnd:.3f} {audioTmp} -loglevel panic"
    subprocess.call(command, shell=True)
    _, audio = wavfile.read(audioTmp)
    command = f"ffmpeg -y -i {cropFile}t.avi -i {audioTmp} -threads {args.nDataLoaderThread} -c:v copy -c:a copy {cropFile}.avi -loglevel panic"
    subprocess.call(command, shell=True)
    os.remove(cropFile + 't.avi')
    return {'track': track, 'proc_track': dets}

def extract_MFCC(file, outPath):
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio, sr)
    featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
    numpy.save(featuresPath, mfcc)

def evaluate_network(files, args, batch_size=8):
    """
    Evaluate the network on multiple video files in parallel.
    
    Args:
        files: List of video files to process
        args: Command line arguments
        batch_size: Number of concurrent tracks to process
    
    Returns:
        List of scores for each track
    """
    # Print only at start and end of processing to avoid interrupting GPU operations
    print(f"Starting to process {len(files)} tracks in parallel...")
    start_time = time.time()
    
    # Load model once
    s = ASD()
    s.loadParameters(args.pretrainModel)
    s.eval()
    
    # Prepare results array
    all_scores = [None] * len(files)
    
    # Define durations once for all tracks
    duration_set = sorted(set([1, 2, 3, 4, 5, 6]))
    
    # Create a lock for GPU operations to avoid contention
    gpu_lock = threading.Lock()
    
    def process_track(idx, file):
        """Process a single track and return its scores"""
        # Extract track filename
        file_name = os.path.splitext(file.split('/')[-1])[0]
        
        # Load audio
        _, audio = wavfile.read(os.path.join(args.pycropPath, file_name + '.wav'))
        audio_feature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        
        # Load video
        video = cv2.VideoCapture(os.path.join(args.pycropPath, file_name + '.avi'))
        video_feature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
                video_feature.append(face)
            else:
                break
        video.release()
        
        video_feature = numpy.array(video_feature)
        length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0])
        audio_feature = audio_feature[:int(round(length * 100)), :]
        video_feature = video_feature[:int(round(length * 25)), :, :]
        
        track_scores = []
        
        # Process each duration
        for duration in duration_set:
            batch_size_internal = int(math.ceil(length / duration))
            duration_scores = []
            
            # Use GPU with lock to avoid contention
            with gpu_lock:
                with torch.no_grad():
                    # Add mixed precision for faster GPU operations
                    with autocast():
                        for i in range(batch_size_internal):
                            input_a = torch.FloatTensor(audio_feature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(0).cuda()
                            input_v = torch.FloatTensor(video_feature[i * duration * 25:(i + 1) * duration * 25, :, :]).unsqueeze(0).cuda()
                            
                            # Forward pass
                            embed_a = s.model.forward_audio_frontend(input_a)
                            embed_v = s.model.forward_visual_frontend(input_v)
                            out = s.model.forward_audio_visual_backend(embed_a, embed_v)
                            score = s.lossAV.forward(out, labels=None)
                            duration_scores.extend(score)
            
            track_scores.append(duration_scores)
        
        # Calculate final scores
        final_scores = numpy.round((numpy.mean(numpy.array(track_scores), axis=0)), 1).astype(float)
        return final_scores
    
    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(files), 8)) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_track, idx, file): idx for idx, file in enumerate(files)}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_scores[idx] = future.result()
            except Exception as e:
                print(f"Error processing track {idx}: {e}")
                # Set error tracks to empty scores
                all_scores[idx] = []
    
    # Final timing information
    end_time = time.time()
    print(f"Finished processing {len(files)} tracks in {end_time - start_time:.2f} seconds")
    
    return all_scores

def evaluate_col_ASD(tracks, scores, args):
    txtPath = args.videoFolder + '/col_labels/fusion/*.txt'
    predictionSet = {}
    for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
        predictionSet[name] = [[], []]
    dictGT = {}
    txtFiles = glob.glob(f"{txtPath}")
    for file in txtFiles:
        lines = open(file).read().splitlines()
        idName = file.split('/')[-1][:-4]
        for line in lines:
            data = line.split('\t')
            frame = int(int(data[0]) / 29.97 * 25)
            x1 = int(data[1])
            y1 = int(data[2])
            x2 = int(data[1]) + int(data[3])
            y2 = int(data[2]) + int(data[3])
            gt = int(data[4])
            if frame in dictGT:
                dictGT[frame].append([x1, y1, x2, y2, gt, idName])
            else:
                dictGT[frame] = [[x1, y1, x2, y2, gt, idName]]
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for _ in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)])
            faces[frame].append({'track': tidx, 'score': float(s), 's': track['proc_track']['s'][fidx], 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        if fidx in dictGT:
            for gtThisFrame in dictGT[fidx]:
                faceGT = gtThisFrame[0:4]
                labelGT = gtThisFrame[4]
                idGT = gtThisFrame[5]
                ious = []
                for face in faces[fidx]:
                    faceLocation = [int(face['x'] - face['s']) // 2, int(face['y'] - face['s']) // 2, int(face['x'] + face['s']) // 2, int(face['y'] + face['s']) // 2]
                    iou = bb_intersection_over_union(faceLocation, faceGT, evalCol=True)
                    if iou > 0.5:
                        ious.append([iou, round(face['score'], 2)])
                if ious:
                    ious.sort()
                    labelPredict = ious[-1][1]
                else:
                    labelPredict = 0
                predictionSet[idGT][0].append(labelPredict)
                predictionSet[idGT][1].append(labelGT)
    names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas']
    names.sort()
    F1s = 0
    for i in names:
        scores = numpy.array(predictionSet[i][0])
        labels = numpy.array(predictionSet[i][1])
        scores = numpy.int64(scores > 0)
        F1 = f1_score(labels, scores)
        ACC = accuracy_score(labels, scores)
        if i != 'abbas':
            F1s += F1
            print(f"{i}, ACC:{100 * ACC:.2f}, F1:{100 * F1:.2f}")
    print(f"Average F1:{100 * (F1s / 5):.2f}")

def main():
    # Define the paths first, just like in the original main() function
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    
    # Create directories in one go
    dirs = [args.pyaviPath, args.pyframesPath, args.pyworkPath, args.pycropPath]
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    
    # Extract video and audio in parallel processes
    video_proc = None
    
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    if args.duration == 0:
        video_cmd = f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -threads {args.nDataLoaderThread} -async 1 -r 25 {args.videoFilePath} -loglevel panic"
    else:
        video_cmd = f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -threads {args.nDataLoaderThread} -ss {args.start:.3f} -to {args.start + args.duration:.3f} -async 1 -r 25 {args.videoFilePath} -loglevel panic"
    
    video_proc = subprocess.Popen(video_cmd, shell=True)
    
    # Start frame extraction after video processing completes
    video_proc.wait()
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Extract the video and save in {args.videoFilePath}\n")
    
    # Extract audio and frames in parallel
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    audio_cmd = f"ffmpeg -y -i {args.videoFilePath} -qscale:a 0 -ac 1 -vn -threads {args.nDataLoaderThread} -ar 16000 {args.audioFilePath} -loglevel panic"
    frames_cmd = f"ffmpeg -y -i {args.videoFilePath} -qscale:v 2 -threads {args.nDataLoaderThread} -f image2 {os.path.join(args.pyframesPath, '%06d.jpg')} -loglevel panic"
    
    audio_proc = subprocess.Popen(audio_cmd, shell=True)
    frames_proc = subprocess.Popen(frames_cmd, shell=True)
    
    # Wait for audio and frames extraction to complete
    audio_proc.wait()
    frames_proc.wait()
    
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Extract the audio and save in {args.audioFilePath}\n")
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Extract the frames and save in {args.pyframesPath}\n")
    
    # Run scene detection and face detection in parallel if possible
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        scene_future = executor.submit(scene_detect, args)
        faces_future = executor.submit(inference_video, args)
        
        # Get results from futures
        scene = scene_future.result()
        faces = faces_future.result()
    
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Scene detection and face detection completed\n")
    
    # Track faces efficiently
    allTracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Face track and detected {len(allTracks)} tracks\n")
    
    # Crop faces in parallel using ThreadPoolExecutor
    vidTracks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.nDataLoaderThread) as executor:
        # Create a list of tuples (index, track) for mapping
        track_with_idx = [(idx, track) for idx, track in enumerate(allTracks)]
        
        # Submit all crop_video tasks to the executor
        futures = [executor.submit(crop_video, args, track, os.path.join(args.pycropPath, f'{idx:05d}')) 
                  for idx, track in track_with_idx]
        
        # Collect results as they complete using tqdm for progress tracking
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            vidTracks.append(future.result())
    
    savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Face Crop and saved in {args.pycropPath} tracks\n")
    
    # Evaluate network with optimized function
    files = glob.glob(f"{args.pycropPath}/*.avi")
    files.sort()
    scores = evaluate_network(files, args)
    savePath = os.path.join(args.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(scores, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Scores extracted and saved in {args.pyworkPath}\n")
    
    # Final evaluation or visualization
    if args.evalCol:
        evaluate_col_ASD(vidTracks, scores, args)
    else:
        reference_image_folder = "users/"
        reference_encodings, reference_user_ids = load_reference_encodings(reference_image_folder)
        visualization(vidTracks, scores, args, reference_encodings, reference_user_ids)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")