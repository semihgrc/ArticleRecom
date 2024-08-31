import numpy as np


class VectorUtils:
    @staticmethod
    def normalize_vector(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def average_normalized_vectors(vector1, vector2):
        normalized_vector1 = VectorUtils.normalize_vector(vector1)
        normalized_vector2 = VectorUtils.normalize_vector(vector2)
        return (normalized_vector1 + normalized_vector2) / 2

    @staticmethod
    def update_user_vector(user_vector,sci_user_vector, article_vector, sci_article_vector):
        new_ft_vector = VectorUtils.average_normalized_vectors(user_vector, article_vector)
        new_sb_vector = VectorUtils.average_normalized_vectors(sci_user_vector, sci_article_vector)
        return new_ft_vector, new_sb_vector


