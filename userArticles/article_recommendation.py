import numpy as np
from bson import ObjectId
from flask import Flask, jsonify
from pymongo import MongoClient

app = Flask(__name__)

class ArticleRecommendation:
    def __init__(self):
        self.mongo_client = MongoClient('localhost', 27017)
        self.db = self.mongo_client['vectors']
        self.user_vector_collection = self.db['user_vectors']
        self.article_vector_collection_ft = self.db['article_vectorsFT']
        self.article_vector_collection_sb = self.db['article_vectorsSB']

    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def recommend_articles(self, user_id):
        user_vector_doc = self.user_vector_collection.find_one({'_id': ObjectId(user_id)})
        if not user_vector_doc:
            return 'User vector not found!', 404

        user_vector_ft = user_vector_doc.get('vector')
        user_vector_sb = user_vector_doc.get('sci_vector')

        article_vectors_ft = {}
        for article_doc in self.article_vector_collection_ft.find():
            article_id = article_doc['article_id']
            article_vector = article_doc['vector']
            article_vectors_ft[article_id] = article_vector

        article_vectors_sb = {}
        for article_doc in self.article_vector_collection_sb.find():
            article_id = article_doc['article_id']
            article_vector = article_doc['vector']
            article_vectors_sb[article_id] = article_vector

        similarity_scores_ft = {}
        for article_id, article_vector in article_vectors_ft.items():
            similarity_scores_ft[article_id] = self.cosine_similarity(user_vector_ft, article_vector)
        top_articles_ft = sorted(similarity_scores_ft.items(), key=lambda x: x[1], reverse=True)[:5]

        similarity_scores_sb = {}
        for article_id, article_vector in article_vectors_sb.items():
            similarity_scores_sb[article_id] = self.cosine_similarity(user_vector_sb, article_vector)
        top_articles_sb = sorted(similarity_scores_sb.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'ft_top_articles': top_articles_ft,
            'sb_top_articles': top_articles_sb
        }

