import os
import numpy as np
from transformers import AutoTokenizer, AutoModel


class SciUserVectorizer:
    def __init__(self, model_name='allenai/scibert_scivocab_uncased'):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        self.model = AutoModel.from_pretrained(model_name, force_download=True)

    def generate_user_vector_from_interests(self, interests):
        vectors = []
        for interest in interests:
            inputs = self.tokenizer(interest, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            vector = np.array(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
            vectors.append(vector)

        user_vector = np.mean(vectors, axis=0)
        return user_vector.tolist()
