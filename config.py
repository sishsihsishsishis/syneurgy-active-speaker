import os
from dotenv import load_dotenv

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_env()
        return cls._instance

    def _load_env(self):
        load_dotenv(".env")
        self.REDIS_HOST = os.getenv("REDIS_HOST")
        self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
        self.SYNC_SERVER = os.getenv("SYNC_SERVER")
        self.REDIS_PORT = os.getenv("REDIS_PORT")
        self.S3_NAME = os.getenv("S3_NAME")
        self.ENV_NAME = os.getenv("ENV_NAME")
        self.AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
        self.AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
        self.AWS_REGION = os.getenv("AWS_REGION")

cfg = Config()
