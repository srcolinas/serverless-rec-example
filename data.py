import os
import shutil
import zipfile

import requests
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix

URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = "ml-100k.zip"
CONTENTS_DIR = "ml-100k"

def maybe_download(url, fpath):
    with open(fpath, "wb") as file:
        print("...downloading to {}".format(fpath))
        response = requests.get(url)
        file.write(response.content)
    print("download complete!")


def zipped_csv_to_df(zip_fpath, csv_fpath, **kwargs):
    with zipfile.ZipFile(zip_fpath) as z:
        
        with z.open(csv_fpath) as f:
            df = pd.read_csv(f, **kwargs)

    return df

def get_items_name():
    csv_fpath = CONTENTS_DIR + "/u.item"
    
    names = ("movie id,movie title,release date,video release date,"
            +"IMDb URL,unknown,Action,Adventure,Animation,Children's,"
            +"Comedy,Crime,Documentary,Drama,Fantasy,Film-Noir,Horror,"
            +"Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western")
            
    kwargs = dict(
        sep='|', header=None,
        names=names.split(','), encoding='latin-1'
    )
        
    df = zipped_csv_to_df(ZIP_PATH, csv_fpath, **kwargs)
    return df['movie title'].values


def convert_ratings_df_to_matrix(
        df, shape, columns="user id,item id,rating".split(',')):
    data = df[columns].values
    users = data[:, 0] - 1 # correct for zero index
    items = data[:, 1] - 1 # correct for zero index
    values = data[:, 2]
    return coo_matrix((values, (users, items)), shape=shape).toarray()


def get_interaction_matrix():
    maybe_download(URL, ZIP_PATH)
    csv_fpath = CONTENTS_DIR + "/u.data"
    df = zipped_csv_to_df(
        ZIP_PATH, csv_fpath, sep='\t',
        header=None, names="user id,item id,rating,timestamp".split(',')
    )
    return convert_ratings_df_to_matrix(df, shape=(943, 1682)).astype(np.float64)


