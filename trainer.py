import argparse
import pickle

import boto3
from sklearn.decomposition import NMF

import data

def train_and_export(n_components, s3_fpath):

    nmf = NMF(n_components)
    interactions = data.get_interaction_matrix()
    nmf.fit(interactions)

    user_factors = nmf.transform(interactions)
    item_factors = nmf.components_

    item_names = data.get_items_name()
    relevant_info = {'user_factors': user_factors, 'item_factors': item_factors,
        'item_names': item_names 
    }

    bucket_name, key = s3_fpath.split('/', 1)
    s3 = boto3.resource('s3')
    s3.Object(bucket_name, key).put(Body=pickle.dumps(relevant_info))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-components', default=30, type=int)
    parser.add_argument('-t', '--target', default='srcolinas-recsys/models/model.pkl')

    args = parser.parse_args()

    train_and_export(args.n_components, args.target)