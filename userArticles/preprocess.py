# preprocess.py

import os
import string
from nltk.corpus import stopwords


class TextPreprocessor:
    def __init__(self):
        pass

    def preprocess_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
        return text

    def preprocess_files(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        files = os.listdir(input_folder)
        for file_name in files:
            with open(os.path.join(input_folder, file_name), 'r', encoding='utf-8') as file:
                text = file.read()
                preprocessed_text = self.preprocess_text(text)
                with open(os.path.join(output_folder, file_name), 'w', encoding='utf-8') as output_file:
                    output_file.write(preprocessed_text)
