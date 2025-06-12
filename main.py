import redis
import subprocess
import time
import logging
import base
import csv
import re
import csv
import requests
import os
import shutil
from config import cfg

def download_user_avatars(meeting_id, max_users=30):
    users_folder = "users"
    
    # Recreate the users folder
    if os.path.exists(users_folder):
        shutil.rmtree(users_folder)
    os.makedirs(users_folder)
    
    for i in range(max_users):
        user_name = f"user{str(i).zfill(2)}"
        endpoint = f"http://{cfg.SYNC_SERVER}:8080/meeting/img/{meeting_id}?userName={user_name}"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            logging.info(f"Got response for {user_name}: {response.status_code}")
            print(response)
            if response.status_code == 404:
                print(f"No image found for {user_name}, stopping.")
                break
            elif response.status_code == 200:
                with open(os.path.join(users_folder, f"{user_name}.jpg"), 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded {user_name}.jpg")
            else:
                print(f"Unexpected status code {response.status_code} for {user_name}")
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {user_name}: {e}")
        except Exception as e:
            print(f"An error occurred for {user_name}: {e}")
            
def fetch_and_write_to_csv(endpoint_url, output_file):
    # Fetch JSON data from the endpoint
    response = requests.get(endpoint_url)
    
    # Print the response content
    print("Response content:")
    print(response.text)
    
    # Parse JSON data
    data = response.json().get('data', [])  # Access the 'data' field and provide an empty list as default value if 'data' is not present

    # Open the output CSV file for writing
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # Write the header row
        writer.writerow(["Speaker", "Start", "End", "Sentence", "Emotion", "DialogueAct"])

        # Process each item in the JSON data
        for idx, item in enumerate(data, start=1):
            try:
                speaker = item.get("speaker")
                start = item.get("starts")
                end = item.get("ends")
                sentence = item.get("sentence")
                emotion = item.get("emotion")
                dialogue_act = item.get("dialogue")

                # Validate mandatory fields
                if None in (speaker, start, end, sentence, emotion, dialogue_act):
                    raise ValueError("One or more mandatory fields are missing or None")

                # Write the extracted fields to the CSV file
                writer.writerow([speaker, start, end, sentence, emotion, dialogue_act])
            except Exception as e:
                print(f"Error processing item {idx}: {item}")
                print(f"Error details: {e}")

if __name__ == '__main__':
    IDLE_COUNT = 0
    while True:
        # get msg from redis
        Z_POP_KEY = 'video-speaker-detector'
        Z_INFO_SEP = ':'

        if IDLE_COUNT > 10: 
            try:
                # This command will shut down the machine (terminate or stop it based on instance settings)
                subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
            except subprocess.CalledProcessError as e:
                print("Failed to execute shutdown command:", e)
            break

        resource_info = base.redis_instance.zpopmin(Z_POP_KEY)
        # [(b'meeting146:test/video/gitlab_video_2.mp4:0.6', 1709577847.4095476)]

        if len(resource_info) < 1:
            IS_LOCKED = 0
            logging.info('empty redis')
            IDLE_COUNT += 1
            time.sleep(20);
        else:
            try: 
                # parse s3 path
                resource_info = str(resource_info[0][0])[2:-1]
                resource_info = base.remove_quotes(resource_info)
                resource_infos = resource_info.split(Z_INFO_SEP)
                meeting_id = resource_infos[0]
                s3_file_path = resource_infos[1]
                s3_file_name = s3_file_path.split("/")[2]
                logging.info(f"video detected {resource_info} {s3_file_name} {meeting_id}")

                # Download from s3
                local_cached_filename = base.download_resource(s3_file_path, './media')

                # Download from s3
                base.download_s3_file(f"test/{meeting_id}/nlp_result.txt", './data/transcript.txt')

                id_ = re.findall('\d+', meeting_id)[0]
                
                # fetch_and_write_to_csv(endpoint_url, output_file)
                download_user_avatars(id_)
                
                process_start_time = time.time()
                # start the video processor
                args = ['--videoName', local_cached_filename.split('.')[0], '--videoFolder', 'media']
                # TODO: Change the path to actual transcript path
                command = ['python3', 'run.py'] + args
                logging.info(f"processing... {local_cached_filename}")

                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                # NORMAL LOGGING
                # stdout, stderr = process.communicate()
                # # logging.info(stdout)
                # logging.error(stderr)

                # # POLLING LOGS
                while True:
                    nextline = process.stdout.readline()
                    if process.poll() is not None and not nextline:
                        break
                    print(nextline.decode('utf-8'), end='')
                process.wait()

                if process.returncode == 0:
                    # start the subprocess
                    # args = ['--videoName', local_cached_filename, '--videoFolder', 'media/input_src']
                    command = ['python3', 'speaker_anomaly.py']
                    logging.info(f"running export csv... {local_cached_filename}")

                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                    # NORMAL LOGGING
                    # stdout, stderr = process.communicate()
                    # # logging.info(stdout)
                    # logging.error(stderr)
                    
                    while True:
                        nextline = process.stdout.readline()
                        if process.poll() is not None and not nextline:
                            break
                        print(nextline.decode('utf-8'), end='')
                    process.wait()

                    if process.returncode == 0:
                        # start the subprocess
                        # args = ['--videoName', local_cached_filename, '--videoFolder', 'media/input_src']
                        command = ['python3', 'export_csv.py']
                        logging.info(f"running export csv... {local_cached_filename}")

                        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                        # POLLING LOGS
                        while True:
                            nextline = process.stdout.readline()
                            if process.poll() is not None and not nextline:
                                break
                            print(nextline.decode('utf-8'), end='')
                        process.wait()
                        
                        process_finish_time = time.time()
                        if process.returncode == 1:
                            logging.error(f"Run.py failed...")

                        if process.returncode == 0:
                            logging.info(f"processed in {process_finish_time - process_start_time}s")
                            local_csv_out_file_path = 'exports/active_speakers.csv'

                            base.upload_file_to_endpoint(meeting_id, local_csv_out_file_path)
                            base.upload_timestamps(meeting_id, process_start_time, process_finish_time)
                            
            except Exception as e:
                log = traceback.format_exc()
                logging.error(f"Error during processing: {log}")
                
            finally:
                os.system('bash -ic "uploadlogs >> logs.txt"')
                s3_key = f"logs/{meeting_id}/speaker_{time.strftime('%m-%d-%H-%M')}.txt"
                base.upload_resource("./logs.txt", s3_key, cfg.S3_NAME)
