# fasttext_train.py

import os

from gensim.models import FastText


class FastTextTrainer:
    def __init__(self):
        pass

    @staticmethod
    def train_model():
        file_names = os.listdir('PreprocessedDatasetTxt')
        sentences = []
        for file_name in file_names:
            with open(os.path.join('PreprocessedDatasetTxt', file_name), 'r', encoding='utf-8') as file:
                text = file.read()
                sentences.append(text.split())
        model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

        model.save('fasttextModel.model')

        return 'FastText model trained and saved successfully!'
