import os
import numpy as np
from gensim.models import FastText
from pymongo import MongoClient
import natsort
class FastTextVectorizer:
    def __init__(self, model_path):
        self.model = FastText.load(model_path)
        self.mongo_client = MongoClient('localhost', 27017)
        self.db = self.mongo_client['vectors']
        self.collection = self.db['article_vectorsFT']

    def generate_and_save_vectors(self, directory):
        file_names = os.listdir(directory)
        file_names = natsort.natsorted(file_names)
        for file_name in file_names[:2000]:
            with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as file:
                content = file.read()
                words = content.split()
                vectors = [self.model.wv[word] for word in words if word in self.model.wv]
                if vectors:
                    article_vector = np.mean(vectors, axis=0)
                    article_data = {
                        'article_id': file_name.split('.')[0],
                        'content': content,
                        'vector': article_vector.tolist()
                    }
                    self.collection.insert_one(article_data)
        self.mongo_client.close()
        return 'First 2000 vectors generated and saved to MongoDB successfully!'














