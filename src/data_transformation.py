from src.common import get_genres
import pandas as pd

class DataTransformer:
    @staticmethod
    def transform_data(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        transformed_movies_df: pd.DataFrame = DataTransformer.transform_movies(movies_df)
        transformed_ratings_df: pd.DataFrame  = DataTransformer.transform_ratings(ratings_df)
        merged_df: pd.DataFrame = pd.merge(transformed_movies_df, transformed_ratings_df, on='movieId', how='left')
        merged_df = DataTransformer.add_rating_percentiles(merged_df)
        movie_rec_df = merged_df.drop(columns=['title', 'year', 'rating'])
        movie_rec_df.set_index('movieId', inplace=True)

        return movie_rec_df

    @staticmethod
    def transform_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
        movies_df['title'] = movies_df['title'].str.replace(r'\((\d{4})\)', '', regex=True)
        movies_df['genres'] = movies_df['genres'].str.split('|')
        for genre in get_genres():
            movies_df[genre] = movies_df['genres'].apply(lambda x: 1 if genre in x else 0)
        movies_df.drop(columns=['genres'], inplace=True)
        return movies_df

    @staticmethod
    def transform_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
        ratings_df.drop(columns=['timestamp'], inplace=True)
        ratings_df_to_merge = ratings_df.drop(columns=['userId'])
        ratings_df_to_merge = ratings_df_to_merge.groupby('movieId')['rating'].mean().reset_index()
        return ratings_df_to_merge

    @staticmethod
    def add_rating_percentiles(merged_df: pd.DataFrame) -> pd.DataFrame:
        percentiles = merged_df['rating'].quantile([0.25, 0.75]).values.tolist()
        merged_df['low_rating'] = merged_df['rating'].apply(lambda x: 1 if x <= percentiles[0] else 0)
        merged_df['medium_rating'] = merged_df['rating'].apply(lambda x: 1 if percentiles[0] < x <= percentiles[1] else 0)
        merged_df['high_rating'] = merged_df['rating'].apply(lambda x: 1 if percentiles[1] < x else 0)
        return merged_df