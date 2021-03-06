import functools
import pickle

import boto3
from flask import Flask
import numpy as np

BUCKET_NAME = 'srcolinas-recsys'
REGION_NAME = 'us-east-1'
S3 = boto3.resource('s3')

app = Flask(__name__)

class FactorsRecommender:

    def __init__(self, item_names, user_factors, item_factors):
        
        self.item_names = item_names
        self.user_factors = user_factors
        self.item_factors = item_factors

    def recommend(self, user_id, top_k=5,
                    return_scores=False, return_names=True):

        scores, indices = self._user_ranking(user_id, top_k=top_k)
    
        return self._format_output(scores, indices,
                        return_scores=return_scores, return_names=return_names) 

    def _format_output(self, scores, indices,
                        return_scores=False, return_names=True):

        best_items = indices
        if return_names:
            item_names = self.item_names
            if item_names is not None:
                best_items = item_names[indices].tolist()

        if return_scores:
            best_scores = scores
            best_items = list(zip(best_items, best_scores))

        return best_items

    def _user_ranking(self, user_id, top_k=5):

        scores = np.dot(self.user_factors[user_id, None], self.item_factors)[0]
       
        indices = np.argsort(scores)[::-1]
        indices.tolist()

        indices = indices[:top_k]
        scores = scores[indices].tolist()

        return scores, indices


@functools.lru_cache()
def load_recommender():
    
    key = 'models/model.pkl'
    # Load model from S3 bucket
    response = S3.Object(bucket_name=BUCKET_NAME, key=key).get()
    
    # Load pickle model
    bytes_ = response['Body'].read()
    dict_ = pickle.loads(bytes_)
    print(dict_)

    rec = FactorsRecommender(
        dict_['item_names'], dict_['user_factors'], dict_['item_factors'])     
    
    return rec


@app.route('/')
def hello():
    return about()

@app.route('/about')
def about():
    return 'Welcome to the recommendations microservice'

@app.route('/recommend/<int:user_id>')
def recommend_to_user(user_id):
       
    recommender = load_recommender()
    recommendations = recommender.recommend(user_id)
    return str(recommendations)
    
if __name__ == '__main__':
    app.run()