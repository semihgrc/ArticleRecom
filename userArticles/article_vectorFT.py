import os
import re
from pymongo import MongoClient
import natsort

class ArticleVectorFT:
    def __init__(self, db_name='vectors', collection_name='article_vectorsFT'):
        self.mongo_client = MongoClient('localhost', 27017)
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]

    def extract_and_update_titles(self, directory, num_articles=2000):
        file_names = os.listdir(directory)
        file_names = natsort.natsorted(file_names)

        for idx, file_name in enumerate(file_names[:num_articles]):
            file_path = os.path.join(directory, file_name)
            title = self.extract_title_from_file(file_path)
            if title:
                article_id = file_name.split('.')[0]
                self.collection.update_one(
                    {'article_id': article_id},
                    {'$set': {'article_name': title}}
                )

        self.mongo_client.close()
        return f"First {num_articles} FastText article titles extracted and updated in MongoDB successfully!"

    def extract_title_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            match = re.search(r'--T\s*(.*?)\s*--A', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None
