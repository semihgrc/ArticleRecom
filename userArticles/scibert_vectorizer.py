import os
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pymongo import MongoClient
import natsort


class SciBERTVectorizer:
    def __init__(self, model_name='allenai/scibert_scivocab_uncased'):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        self.model = AutoModel.from_pretrained(model_name, force_download=True)
        self.mongo_client = MongoClient('localhost', 27017)
        self.db = self.mongo_client['vectors']
        self.collection = self.db['article_vectorsSB']

    def generate_and_save_vectors(self, directory):
        try:
            file_names = os.listdir(directory)
            file_names = natsort.natsorted(file_names)

            for file_name in file_names[:2000]:
                with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as file:
                    content = file.read()
                    inputs = self.tokenizer(content, return_tensors='pt', padding=True, truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    article_vector = np.array(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
                    article_data = {
                        'article_id': file_name.split('.')[0],
                        'content': content,
                        'vector': article_vector.tolist()
                    }
                    self.collection.insert_one(article_data)

            self.mongo_client.close()
            return 'First 2000 vectors generated and saved to MongoDB successfully!'
        except Exception as e:
            return f'An error occurred: {e}'
