# mongo_helpers.py
from bson import ObjectId
from pymongo import MongoClient
from bson.json_util import dumps

# MongoDB bağlantısını başlat
client = MongoClient('mongodb://localhost:27017/')
db = client['vectors']
collection = db['article_vectorsFT']

def fetch_all_articles():
    articles = collection.find({}, {'_id': 1, 'article_id': 1, 'article_name': 1})
    articles_list = list(articles)
    return articles_list

def fetch_article_by_id(article_id):
    article = collection.find_one({'_id': ObjectId(article_id)})
    if article:
        return dumps(article)
    else:
        return None
