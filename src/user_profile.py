import pandas as pd
from src.common import get_genres
import numpy as np

class UserProfileCreator:
    @staticmethod
    def build_user_profile(movie_rec_df: pd.DataFrame, user_ratings: pd.DataFrame) -> pd.Series:
        user_ratings['rating'] = user_ratings['rating'].apply(lambda x: x*x)
        user_movies = movie_rec_df.loc[user_ratings['movieId']]

        # multiply the test_user_movies elements by the test_user_ratings elements
        profile = user_movies.T.dot(user_ratings['rating'].values)
        profile[['low_rating', 'medium_rating', 'high_rating']] = profile[['low_rating', 'medium_rating', 'high_rating']] / (2 *profile[['low_rating', 'medium_rating', 'high_rating']].sum())

        # normalize all other values
        norm = profile.sum()
        for genre in get_genres():
            profile[genre] = profile[genre] / norm

        return profile