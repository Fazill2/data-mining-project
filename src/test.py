import numpy as np
import pandas as pd

class RankingTest:
    @staticmethod
    def split_ratings_into_train_test(ratings: pd.DataFrame, k=5, seed=42) -> (pd.DataFrame, pd.DataFrame, list):
        """
        Splits the ratings into train and test sets, k ratings of each user are in test set.

        Parameters:
        rankings (pd.DataFrame): All ratings
        n (float): The ratio for splitting the data.

        Returns:
        tuple: A tuple containing the train and test sets.
        """
        rng = np.random.default_rng(seed)

        # Shuffle within each user and mark k items for test
        df = ratings.copy()

        # Assign a random number for sorting within each user
        df['rand'] = rng.random(len(df))

        # Rank rows within each user based on the random number
        df['rank'] = df.groupby('userId')['rand'].rank(method='first', ascending=False)

        # Split based on rank
        test = df[df['rank'] <= k].drop(columns=['rand', 'rank'])
        train = df[df['rank'] > k].drop(columns=['rand', 'rank'])

        return train, test

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

    @staticmethod
    def calculate_spearman_corr(user_ratings: pd.DataFrame, ranking: pd.Series) -> float:
        """
        Calculates the Spearman correlation coefficient between the user ratings and the ranking.

        Parameters:
        user_ratings (pd.Series): The user ratings.
        ranking (pd.DataFrame): The ranked movies.

        Returns:
        float: The Spearman correlation coefficient.
        """
        # user_highest_rating = user_ratings['rating'].max()
        # threshold = 4 if user_highest_rating >= 4 else user_highest_rating - 0.5
        merged = pd.merge(user_ratings, ranking, on='movieId', suffixes=('_rating', '_ranking'))
        # merged['rating'] = merged['rating'] > threshold
        # merged.at[merged.index[0], 'rating'] = True
        # merged.at[merged.index[-1], 'rating'] = False

        spearman_corr = merged['rating'].corr(merged['score'], method='spearman')
        return spearman_corr

    @staticmethod
    def calculate_pearson_corr(user_ratings: pd.DataFrame, ranking: pd.Series) -> float:
        """
        Calculates the Pearson correlation coefficient between the user ratings and the ranking.

        Parameters:
        user_ratings (pd.Series): The user ratings.
        ranking (pd.DataFrame): The ranked movies.

        Returns:
        float: The Pearson correlation coefficient.
        """
        # user_highest_rating = user_ratings['rating'].max()
        # threshold = 4 if user_highest_rating >= 4 else user_highest_rating - 0.5
        merged = pd.merge(user_ratings, ranking, on='movieId', suffixes=('_rating', '_ranking'))
        # merged['rating'] = merged['rating'] > threshold
        # merged.at[merged.index[0], 'rating'] = True
        # merged.at[merged.index[-1], 'rating'] = False
        pearson_corr = merged['rating'].corr(merged['score'], method='pearson')
        return pearson_corr

    @staticmethod
    def calculate_rr(user_ratings: pd.DataFrame, ranking: pd.Series) -> float:
        """
        Calculates the Reciprocal Rank (RR) for the given user ratings and ranking.

        Parameters:
        user_ratings (pd.Series): The user ratings.
        ranking (pd.DataFrame): The ranked movies.

        Returns:
        float: The RR score.
        """
        # Calculate RR
        rr = 0
        for i, movie in enumerate(ranking['movieId']):
            if movie in user_ratings['movieId'].values:
                rr = 1 / (i + 1)
                break


        return rr

    @staticmethod
    def calculate_mrr(user_ratings: pd.DataFrame, rankings: dict[str, pd.DataFrame]) -> float:
        """
        Calculates the Mean Reciprocal Rank (MRR) for the given user ratings and rankings.

        Parameters:
        user_ratings (pd.Series): The user ratings.
        rankings (dict[str, pd.DataFrame]): The ranked movies.

        Returns:
        float: The MRR score.
        """
        mrr = 0
        for user, ranking in rankings:
            mrr += RankingTest.calculate_rr(user_ratings[user_ratings['userId'] == user], ranking)

        return mrr / len(rankings)

