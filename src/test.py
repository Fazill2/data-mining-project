import numpy as np
import pandas as pd

class RankingTest:
    @staticmethod
    def split_ratings_into_train_test(ratings: pd.DataFrame, n=0.1) -> (pd.DataFrame, pd.DataFrame, list):
        """
        Splits the ratings into train and test sets based on the specified ratio.

        Parameters:
        rankings (pd.DataFrame): All ratings
        n (float): The ratio for splitting the data.

        Returns:
        tuple: A tuple containing the train and test sets and the selected user IDs.
        """
        selected_user_ids = RankingTest.get_test_user_ids(ratings, n)
        train_set = ratings[ratings['userId'].isin(selected_user_ids) == False]
        test_set = ratings[ratings['userId'].isin(selected_user_ids)]
        return train_set, test_set, selected_user_ids

    @staticmethod
    def get_test_user_ids(ratings: pd.DataFrame, n=0.1) -> list:
        """
        Selects a subset of user IDs for testing based on the specified ratio.

        Parameters:
        ratings (pd.DataFrame): The user ratings.
        n (float): The ratio for selecting user IDs.

        Returns:
        list: A list of selected user IDs.
        """
        user_ids = ratings.index.unique()
        selected_user_ids = np.random.choice(user_ids, size=int(len(user_ids) * n), replace=False)
        return selected_user_ids

    @staticmethod
    def split_user_ratings(user_ratings: pd.DataFrame, n=0.1, k = 10) -> (pd.Series, pd.Series):
        """
        Splits user ratings into train and test sets based on the specified ratio.

        Parameters:
        user_ratings (pd.Series): The user ratings.
        n (float): The ratio for splitting the data.

        Returns:
        tuple: A tuple containing the train and test sets.
        """
        user_ratings = user_ratings.sample(frac=1, random_state=42)
        if len(user_ratings) == 0:
            return pd.Series(), pd.Series()
        # make test set take at lest k items
        if (len(user_ratings) * n) < k:
            n = k / len(user_ratings)
        split_index = int(len(user_ratings) * (1 - n))
        train_set = user_ratings[:split_index]
        test_set = user_ratings[split_index:]
        return train_set, test_set

    @staticmethod
    def calculate_ndcg(user_ratings: pd.DataFrame, ranking: pd.Series, k: int=10) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) for the given user ratings and ranking.

        Parameters:
        user_ratings (pd.Series): The user ratings.
        ranking (pd.DataFrame): The ranked movies.
        k (int): The number of top recommendations to consider.

        Returns:
        float: The NDCG score.
        """
        # Get the top k recommended movies
        recommended_movies = ranking.head(k)

        # Calculate DCG
        dcg = 0
        for i, movie in enumerate(recommended_movies.index):
            if movie in user_ratings['movieId'].values:
                dcg += user_ratings[user_ratings['movieId']==movie]['rating'].values[0] / np.log2(i + 2)

        # Calculate IDCG
        idcg = 0
        ideal_ranking = user_ratings.sort_values(by='rating', ascending=False).head(k)
        for i, movie in enumerate(ideal_ranking['movieId']):
            idcg += ideal_ranking[ideal_ranking['movieId']==movie]['rating'].values[0] / np.log2(i + 2)

        # Calculate NDCG
        ndcg = 0
        try:
            ndcg = dcg / idcg if idcg > 0 else 0
        except Exception as e:
            print(f"Error processing user {e} with NDCG calculation dcg: {dcg}, idcg: {idcg}")
            raise e

        return ndcg