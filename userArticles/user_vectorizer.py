from gensim.models import FastText
import numpy as np


class UserVectorizer:
    def __init__(self, model_path):
        self.model = FastText.load(model_path)

    def generate_user_vector(self, interests):
        vectors = [self.model.wv[interest] for interest in interests]
        user_vector = np.mean(vectors, axis=0)
        return user_vector.tolist()

    def generate_user_vector_from_indices(self, interest_indices):
        vectors = [self.model.wv[index] for index in interest_indices]
        user_vector = np.mean(vectors, axis=0)
        return user_vector.tolist()
