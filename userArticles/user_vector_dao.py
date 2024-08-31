from pymongo import MongoClient

class UserVectorDAO:
    def __init__(self):
        self.mongo_client = MongoClient('localhost', 27017)
        self.db = self.mongo_client['vectors']
        self.collection = self.db['user_vectors']

    def save_user_vector(self, user_id, vector):
        user_vector = {'user_id': user_id, 'vector': vector}
        self.collection.insert_one(user_vector)
