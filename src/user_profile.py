import pandas as pd
from src.common import get_genres
import numpy as np

class UserProfileCreator:
    @staticmethod
    def build_user_profiles(movie_rec_df: pd.DataFrame, user_ratings: pd.DataFrame, norm='l2') -> pd.Series:
        movie_vectors = movie_rec_df.loc[user_ratings['movieId']].reset_index()
        movie_vectors['userId'] = user_ratings['userId'].values
        movie_vectors['rating'] = user_ratings['rating'].values

        for column in movie_rec_df.columns:
            movie_vectors[column] = movie_vectors[column] * movie_vectors['rating']

        user_profiles = movie_vectors.groupby('userId')[movie_rec_df.columns].sum()
        user_profiles = user_profiles.div(user_profiles.sum(axis=1), axis=0)

        return user_profiles