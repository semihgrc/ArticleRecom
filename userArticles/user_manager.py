from bson import ObjectId
from flask import jsonify, make_response
from pymongo import MongoClient
from sci_user_vectorizer import SciUserVectorizer
from user_vectorizer import UserVectorizer


class UserManager:
    def __init__(self):
        self.mongo_client = MongoClient('localhost', 27017)
        self.db = self.mongo_client['vectors']
        self.user_collection = self.db['user_vectors']
        self.vectorizer = UserVectorizer('fasttextModel.model')
        self.sci_vectorizer = SciUserVectorizer('allenai/scibert_scivocab_uncased')

    def authenticate_user(self, username, password):
        user = self.user_collection.find_one({'username': username, 'password': password})
        if user:
            return str(user['_id'])
        else:
            return None

    def get_user_info(self, user_id):
        user_info = self.user_collection.find_one({'_id': ObjectId(user_id)})
        if user_info:
            response_data = {
                'username': user_info['username'],
                'password': user_info['password'],
                'interests': user_info['interests'],
                'vector': user_info['vector']
            }
            return make_response(jsonify(response_data), 200)
        else:
            return make_response(jsonify({'error':'User not found'}), 404)

    def signup_user(self, username, password, first_name, last_name, age, gender, education, interests):
        user_vector = self.vectorizer.generate_user_vector_from_indices(interests)
        sci_user_vector = self.sci_vectorizer.generate_user_vector_from_interests(interests)
        user_data = {
            'username': username,
            'password': password,
            'first_name': first_name,
            'last_name': last_name,
            'age': age,
            'gender': gender,
            'education': education,
            'interests': interests,
            'vector': user_vector,
            'sci_vector': sci_user_vector
        }
        self.user_collection.insert_one(user_data)
        return 'User signed up successfully.'
