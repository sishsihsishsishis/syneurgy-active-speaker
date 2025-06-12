import logging
import re
import sys
import boto3
import requests
import time
import redis
import uuid
import os
import shutil
import traceback
from config import cfg

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %Z')

aws_access_key_id = cfg.AWS_ACCESS_KEY
aws_secret_access_key = cfg.AWS_SECRET_KEY
aws_region = cfg.AWS_REGION

# RETURN THE S3 INSTANCE
def get_s3_instance():
    # init s3
    s3_instance = boto3.client(service_name='s3', region_name='us-west-1', endpoint_url='https://s3.us-west-1.amazonaws.com', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    return s3_instance

# RETURN THE REDIS CLIENT INSTANCE
def get_redis_instance():
    pool = redis.ConnectionPool(host=cfg.REDIS_HOST, password=cfg.REDIS_PASSWORD, port=cfg.REDIS_PORT, db=0)
    redis_instance = redis.StrictRedis(connection_pool=pool)
    return redis_instance

# DOWNLOAD RAW VIDEO FROM S3
def download_resource(resource_path, base_cache_dir, bucket_name=cfg.S3_NAME):
    try:
        logging.info(f"downloading... {resource_path}")
        start = time.time()

        # initialize s3 and download file in cache dir
        s3_instance = get_s3_instance()

        # init s3 cache dir
        if os.path.exists(base_cache_dir):
            if os.path.isdir(base_cache_dir):
                shutil.rmtree(base_cache_dir, ignore_errors=True)
            elif os.path.isfile(base_cache_dir):
                os.remove(base_cache_dir)
        cache_dir = os.path.join(base_cache_dir)

        os.makedirs(cache_dir)
        file_name = uuid.uuid4().hex
        cached_path = os.path.join(cache_dir, '{}.mp4'.format(file_name))
        s3_instance.download_file(bucket_name, resource_path, cached_path)

        finish = time.time()
        logging.info(f"downloaded in {finish - start}s")
        return file_name
    except: 
        log = traceback.format_exc()
        logging.error(log)

# DOWNLOAD AVATAR FROM S3
def download_s3_file(file_path, local_path, bucket_name=cfg.S3_NAME):
    try:
        logging.info(f"Downloading avatar... {file_path}")
        start = time.time()

        # Initialize S3 instance
        s3_instance = get_s3_instance()

        s3_instance.download_file(bucket_name, file_path, local_path)

        finish = time.time()
        logging.info(f"Avatar downloaded successfully to {local_path} in {finish - start}s")
        return local_path

    except Exception:
        logging.error(f"{file_path} doesn't exist")
        return None

# UPLOAD THE OUTPUT VIDEO TO S3
def upload_resource(filename, key, bucket_name=cfg.S3_NAME):
    logging.info('uploading', key)
    start = time.time()

    # initialize s3 and upload file
    s3_instance = get_s3_instance()
    s3_instance.upload_file(filename, bucket_name, key)

    finish = time.time()
    print('uploaded in ', finish - start, 's')
    return filename

# UPLOAD FILES THROUGH ENDPOINT
def upload_file_to_endpoint(meeting_id, file_path):
    try:
        id_ = re.findall('\d+', meeting_id)[0]
        url = f"http://{cfg.SYNC_SERVER}:8080/s3/uploadByMeeting/"
        url += id_

        logging.info(f"Uploading file.. {id_} {url}")
        files = {'file': open(file_path, 'rb')}
        response = requests.post(url, files=files)
        if response.ok:
            logging.info(f"File Upload Success :) \n:::: {response.text}")
        else:
            logging.error(f"File Upload Error :) \n:::: {response.status_code} {response.text}")
        logging.info(response.text)
    except:
        log = traceback.format_exc()
        logging.error(log)

# UPLOAD TIMESTAMPS
def upload_timestamps(meeting_id, start, end):
    try: 
        logging.info(f"Uploading timestamps.. {int(start*1000)} {int(end*1000)}")
        data = {
            "model-emotion-detection": [int(start*1000), int(end*1000)]
        }

        id_ = re.findall('\d+', meeting_id)[0]
        url = f"http://{cfg.SYNC_SERVER}:8080/meeting/analysis-time/{id_}"

        logging.info(f"Uploading file to Endpoint.. {id_} {url}")
        response = requests.post(url, json=data)
        if response.ok:
            logging.info(f"Timestamps Upload Success :) \n:::: {response.text}")
        else:
            logging.error(f"Timestamps Upload Error :) \n:::: {response.status_code} {response.text}")
    except:
        log = traceback.format_exc()
        logging.error(log)

# remove double quotes from the string
def remove_quotes(input_string):
    if input_string.startswith('"') and input_string.endswith('"'):
        # If the string starts and ends with double quotes
        # Remove the first and last character (double quotes)
        logging.info("Redis entry has double quotes :(")
        return input_string[1:-1]
    else:
        # If the string does not start and end with double quotes
        return input_string

# INITIALIZE REDIS AS GLOBAL
redis_instance = get_redis_instance()
