import pandas as pd
import numpy as np

class SpeakerIDMapper:
    def __init__(self, raw_detection_path, transcript_path, output_path):
        self.raw_detection_path = raw_detection_path
        self.transcript_path = transcript_path
        self.output_path = output_path

    def map_speaker_ids_with_overlap_handling(self):
        # Load raw detection data and sort by timestamp
        raw_df = pd.read_csv(self.raw_detection_path)
        raw_df = raw_df.sort_values(by='timestamp')
        raw_df['timestamp'] = raw_df['timestamp'].astype(float)

        # Load transcript data
        transcript_df = pd.read_csv(self.transcript_path, sep='\t', 
                                names=['Speaker', 'Start', 'End', 'Sentence', 'Emotion', 'DialogueAct'])
        transcript_df[['Start', 'End']] = transcript_df[['Start', 'End']].apply(pd.to_numeric, errors='coerce')
        transcript_df = transcript_df.dropna(subset=['Start', 'End'])

        # Use vectorized operations instead of iterating through tuples
        results = []
        used_timestamps = set()
        
        # Convert to numpy for faster operations
        raw_array = raw_df.to_numpy()
        transcript_array = transcript_df[['Speaker', 'Start', 'End']].to_numpy()
        
        # Use vectorized boolean masks for matching
        for idx, raw_row in enumerate(raw_array):
            raw_timestamp = raw_row[1]
            
            # Create boolean masks for matching transcript rows
            matches = (transcript_array[:, 1] <= raw_timestamp) & (raw_timestamp <= transcript_array[:, 2])
            matching_indices = np.where(matches)[0]
            
            if len(matching_indices) != 1:
                continue
                
            match_idx = matching_indices[0]
            speaker_id = transcript_array[match_idx, 0]
            start_time = transcript_array[match_idx, 1]
            end_time = transcript_array[match_idx, 2]
            
            # Check for overlaps with used timestamps - this is hard to vectorize efficiently
            overlap = any(start_time <= used_timestamp <= end_time for used_timestamp in used_timestamps)
            if overlap:
                # Remove any overlapping entries from results
                results = [res for res in results if not (res[1] <= raw_timestamp <= res[2])]
                # Also remove from used timestamps
                used_timestamps = {ts for ts in used_timestamps if not (start_time <= ts <= end_time)}
            
            # Append the result
            results.append([speaker_id, start_time, end_time] + raw_row[2:6].tolist())
            used_timestamps.add(raw_timestamp)
        
        # Create a DataFrame from the results and save to CSV
        results_df = pd.DataFrame(results, columns=['User_ID', 'Start_time', 'End_time', 'X1', 'Y1', 'X2', 'Y2'])
        results_df.to_csv(self.output_path, index=False)

# Define file paths
raw_detection_path = 'exports/corrected_detection.csv'
transcript_path = 'data/transcript.txt'
output_path = 'exports/active_speakers.csv'

# Create an instance of the class and call the method
mapper = SpeakerIDMapper(raw_detection_path, transcript_path, output_path)
mapper.map_speaker_ids_with_overlap_handling()
