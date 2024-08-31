import os
import numpy as np
from bson import ObjectId
from bson.json_util import dumps
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from article_recommendation import ArticleRecommendation
from article_vectorFT import ArticleVectorFT
from article_vectorSB import ArticleVectorSB
from fasttext_train import FastTextTrainer
from fasttext_vectorizer import FastTextVectorizer
from mongo_helpers import fetch_all_articles, fetch_article_by_id
from preprocess import TextPreprocessor
from scibert_vectorizer import SciBERTVectorizer
from user_manager import UserManager
from user_vector_dao import UserVectorDAO
from user_vectorizer import UserVectorizer
from vector_utils import VectorUtils

app = Flask(__name__)
CORS(app)
if __name__ == '__main__':
    app.run(debug=True)

model_name = "allenai/scibert_scivocab_uncased"
FASTTEXT_MODEL_PATH = 'fasttextModel.model'
user_vectorizer = UserVectorizer(FASTTEXT_MODEL_PATH)
user_vector_dao = UserVectorDAO()
article_recommender = ArticleRecommendation()
user_manager = UserManager()
client = MongoClient('mongodb://localhost:27017/')
db = client['vectors']
users_collection = db['user_vectors']
articles_collection = db['article_vectorsFT']
articles_collection_ft = db['article_vectorsFT']
articles_collection_sb = db['article_vectorsSB']
articles_ft_collection = db['article_vectorsFT']
articles_sb_collection = db['article_vectorsSB']


@app.route('/generate_scibert_vectors', methods=['GET'])
def generate_scibert_vectors():
    directory = 'PreprocessedDatasetTxt'
    scibert_vectorizer = SciBERTVectorizer()
    result = scibert_vectorizer.generate_and_save_vectors(directory)
    if not directory:
        return jsonify({'error': 'Directory path is required'}), 400
    if 'An error occurred' in result:
        return jsonify({'error': result}), 500
    return jsonify({'message': result}), 200


@app.route('/generateVectors')
def generate_vectors():
    vectorizer = FastTextVectorizer('fasttextModel.model')
    result = vectorizer.generate_and_save_vectors('PreprocessedDatasetTxt')
    return result


@app.route('/generate_ft_titles', methods=['GET'])
def generate_ft_titles():
    directory = 'DatasetTxt'
    if not os.path.exists(directory):
        return jsonify({'error': 'Directory not found'}), 400

    ft_vectorizer = ArticleVectorFT()
    result = ft_vectorizer.extract_and_update_titles(directory)

    return jsonify({'message': result}), 200


@app.route('/generate_sb_titles', methods=['GET'])
def generate_sb_titles():
    directory = 'DatasetTxt'
    if not os.path.exists(directory):
        return jsonify({'error': 'Directory not found'}), 400

    sb_vectorizer = ArticleVectorSB()
    result = sb_vectorizer.extract_and_update_titles(directory)

    return jsonify({'message': result}), 200


@app.route('/user/<string:user_id>', methods=['GET'])
def get_user(user_id):
    top_articles = article_recommender.recommend_articles(user_id)
    return jsonify(top_articles)


@app.route('/articles', methods=['GET'])
def articles_endpoint():
    articles = fetch_all_articles()
    return jsonify(dumps(articles))


@app.route('/articletext/<string:article_id>', methods=['GET'])
def article_endpoint(article_id):
    article = fetch_article_by_id(article_id)
    if article:
        return article
    else:
        return jsonify({'error': 'Article not found'}), 404


@app.route('/articletext/<string:userId>/<string:article_id>', methods=['POST'])
def update_user_vector(userId, article_id):
    user = users_collection.find_one({'_id': ObjectId(userId)})
    article_ft = articles_ft_collection.find_one({'_id': ObjectId(article_id)})

    if not user or not article_ft:
        return jsonify({'error': 'User or article not found'}), 404

    user_vector = user.get('vector')
    sci_user_vector = user.get('sci_vector')
    article_vector_ft = article_ft.get('vector')

    if not user_vector or not article_vector_ft:
        return jsonify({'error': 'Vector not found for user or article'}), 404
    if not sci_user_vector or not article_vector_ft:
        return jsonify({'error': 'Sci Vector not found for user or article'}), 404

    article_id_str = article_ft.get('article_id')
    article_sb = articles_sb_collection.find_one({'article_id': article_id_str})

    if not article_sb:
        return jsonify({'error': 'SciBERT article not found'}), 404

    article_vector_sb = article_sb.get('vector')

    if not article_vector_sb:
        return jsonify({'error': 'SciBERT vector not found for article'}), 404

    user_vector = np.array(user_vector)
    sci_user_vector = np.array(sci_user_vector)
    article_vector_ft = np.array(article_vector_ft)
    article_vector_sb = np.array(article_vector_sb)

    new_ft_vector, new_sb_vector = VectorUtils.update_user_vector(user_vector, sci_user_vector, article_vector_ft, article_vector_sb)
    users_collection.update_one({'_id': ObjectId(userId)}, {'$set': {'vector': new_ft_vector.tolist(), 'sci_vector': new_sb_vector.tolist()}})

    # Article name and content
    article_name = article_ft.get('article_name', 'No Title')
    article_content = article_ft.get('content', 'No Content')

    return jsonify({
        'message': 'User vectors updated successfully',
        'article_name': article_name,
        'content': article_content
    }), 200



@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'message': 'Please provide username and password.'}), 400
    user_id = user_manager.authenticate_user(username, password)
    if user_id:
        return jsonify({'message': 'Login successful!', 'userId': user_id}), 200
    else:
        return jsonify({'message': 'Invalid username or password.'}), 401


@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    first_name = data.get('firstName')
    last_name = data.get('lastName')
    age = data.get('age')
    gender = data.get('gender')
    education = data.get('education')
    interests = data.get('interests')

    if not username or not password or not first_name or not last_name or not age or not gender or not education or not interests:
        return jsonify({'message': 'Please provide all required fields.'}), 400

    if user_manager.user_collection.find_one({'username': username}):
        return jsonify({'message': 'Username already exists. Please choose another one.'}), 400

    message = user_manager.signup_user(username, password, first_name, last_name, age, gender, education, interests)
    return jsonify({'message': message}), 201

if __name__ == '__main__':
    app.run(debug=True)
