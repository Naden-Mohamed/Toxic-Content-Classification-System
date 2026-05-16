import os
import csv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from configs.config import Config

class Database:
    def __init__(self):
        self.config = Config()
        self.client = None
        self.collection = None

    def connect(self):
            if self.config.COLLECTION_NAME and self.config.DATABASE_NAME and self.client:
                db = self.client[self.config.DATABASE_NAME]
                self.collection = db[self.config.COLLECTION_NAME]

            self.client = MongoClient(self.config.MONGODB_URI, server_api=ServerApi('1'))

            

    def insert_record(self, text, result):
        self.connect()

        document = {
            "text": text,
            "label": result["label"],
            "confidence": result["confidence"]
        }
        if bool(self.collection):
            self.collection.insert_one(document)

    def append_to_csv(self, text, result):
        file_path = self.config.CSV_PATH or "data/toxic_logs.csv"  
        file_exists = os.path.isfile(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["text", "label", "confidence"])

            writer.writerow([text, result["label"], result["confidence"]])