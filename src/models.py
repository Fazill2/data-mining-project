from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from sklearn.cluster import KMeans
from src.user_profile import UserProfileCreator
import numpy as np

class AprioriKmeansRecommender:
    def __init__(self, item_data: pd.DataFrame, min_support=0.5, min_confidence=0.5, k=15, random_state=42):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.item_data = item_data
        self._train_ratings = None
        self.k = k
        self.random_state = random_state
        self._train_profiles = None
        self._train_profiles_clusters = None
        self._centroids = None
        self._cluster_rules = {}
        self._kmeans = None

    def fit(self, train_ratings: pd.DataFrame):
        self._train_ratings = train_ratings
        self._train_profiles = UserProfileCreator.build_user_profiles(self.item_data, self._train_ratings)


        self._kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
        self._kmeans.fit(self._train_profiles)
        self._train_profiles['cluster'] = self._kmeans.labels_

        self._train_profiles_clusters = self._train_profiles.groupby('cluster')

        for cluster_id in range(self.k):
            self._generate_rules_for_cluster(cluster_id)

    def _generate_rules_for_cluster(self, cluster_id):
        cluster_user_ids = self._train_profiles[self._train_profiles['cluster'] == cluster_id].index
        cluster_ratings = self._train_ratings[self._train_ratings['userId'].isin(cluster_user_ids)]

        merged = cluster_ratings.merge(self.item_data, on='movieId')

        liked = merged[merged['rating'] >= 4]

        genre_columns = [col for col in self.item_data.columns if col not in ['movieId', 'userId', 'rating']]
        genre_transactions = liked[genre_columns]

        genre_transactions = genre_transactions.astype(bool)

        frequent_itemsets = apriori(genre_transactions, min_support=self.min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=self.min_confidence)

        self._cluster_rules[cluster_id] = rules

    def recommend(self, user_profile: pd.Series):
        user_profile = user_profile.values.reshape(1, -1)
        cluster = self._centroids.predict(user_profile)

        rules = self._cluster_rules[cluster[0]]

        recommendations = []
        for _, rule in rules.iterrows():
            recommendations.extend(rule['consequents'])

        return list(set(recommendations))

    def recommend_for_user(self, user_ratings: pd.DataFrame, evaluation_ratings: pd.DataFrame = None, top_n=5, penalize_genres=False):
        user_profile = UserProfileCreator.build_user_profiles(self.item_data, user_ratings)
        user_cluster = self._kmeans.predict(user_profile.values.reshape(1, -1))[0]

        rules = self._cluster_rules.get(user_cluster)
        if rules is None or rules.empty:
            return []

        user_highest_rating = user_ratings['rating'].max()
        threshold = 4 if user_highest_rating >= 4 else user_highest_rating - 0.5


        liked_movies = user_ratings[user_ratings['rating'] >= threshold]
        liked_movie_ids = liked_movies['movieId'].tolist()

        liked_genres = self.item_data[self.item_data.index.isin(liked_movie_ids)]
        genre_columns = [col for col in self.item_data.columns if col not in ['movieId', 'userId', 'rating']]
        liked_genres = liked_genres[genre_columns]

        liked_genres = liked_genres.any().index[liked_genres.any().values].tolist()

        matching_rules = []
        for idx, rule in rules.iterrows():
            antecedents = set(rule['antecedents'])
            if antecedents.issubset(liked_genres):
                matching_rules.append((rule['consequents'], rule['confidence']))

        matching_rules = sorted(matching_rules, key=lambda x: -x[1])

        recommended_genres = set()
        conf_map = {}
        for consequents, confidence in matching_rules:
            recommended_genres.update(consequents)
            for genre in consequents:
                conf_map[genre] = conf_map.get(genre, 0) + confidence

        candidate_movies = self.item_data.copy()
        if evaluation_ratings is not None:
            candidate_movies = candidate_movies[candidate_movies.index.isin(evaluation_ratings['movieId'])]
        candidate_movies['score'] = 0
        if penalize_genres:
            candidate_movies['genre_count'] = np.sqrt(candidate_movies[genre_columns].sum(axis=1))
        else:
            candidate_movies['genre_count'] = 1
        for genre in recommended_genres:
            candidate_movies['score'] += conf_map.get(genre, 0) * candidate_movies[genre] / candidate_movies['genre_count']

        already_rated = set(user_ratings['movieId'])
        candidate_movies = candidate_movies[~candidate_movies.index.isin(already_rated)]
        candidate_movies.sort_values(by='score', ascending=False, inplace=True)
        recommendations = candidate_movies.drop_duplicates().head(top_n)
        recommendations['movieId'] = recommendations.index
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations

    def get_rules(self):
        return self._cluster_rules

class AprioriSimplestRecommender:
    def __init__(self, item_data: pd.DataFrame, min_support=0.5, min_confidence=0.5, random_state=42):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.item_data = item_data
        self._train_ratings = None
        self.random_state = random_state
        self._train_profiles = None
        self._train_profiles_clusters = None
        self._centroids = None
        self._cluster_rules = {}
        self._kmeans = None

    def recommend_for_user(self, user_ratings: pd.DataFrame, evaluation_ratings: pd.DataFrame = None, top_n=5, penalize_genres=False):
        merged = user_ratings.merge(self.item_data, on='movieId')
        user_highest_rating = user_ratings['rating'].max()
        threshold = 4 if user_highest_rating >= 4 else user_highest_rating - 0.5
        liked = merged[merged['rating'] >= threshold]
        genre_columns = [col for col in self.item_data.columns if col not in ['movieId', 'userId', 'rating']]
        genre_transactions = liked[genre_columns]

        genre_transactions = genre_transactions.astype(bool)

        frequent_itemsets = apriori(genre_transactions, min_support=self.min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=self.min_confidence)
        final_rules = []
        for idx, rule in rules.iterrows():
            antecedents = set(rule['antecedents'])
            final_rules.append((rule['consequents'], rule['confidence']))
        recommended_genres = set()
        conf_map = {}
        for consequents, confidence in final_rules:
            recommended_genres.update(consequents)
            for genre in consequents:
                conf_map[genre] = conf_map.get(genre, 0) + confidence
        candidate_movies = self.item_data.copy()
        if evaluation_ratings is not None:
            candidate_movies = candidate_movies[candidate_movies.index.isin(evaluation_ratings['movieId'])]
        candidate_movies['score'] = 0
        if penalize_genres:
            candidate_movies['genre_count'] = np.sqrt(candidate_movies[genre_columns].sum(axis=1))
        else:
            candidate_movies['genre_count'] = 1
        for genre in recommended_genres:
            candidate_movies['score'] += conf_map.get(genre, 0) * candidate_movies[genre] / candidate_movies['genre_count']
        already_rated = set(user_ratings['movieId'])
        candidate_movies = candidate_movies[~candidate_movies.index.isin(already_rated)]
        candidate_movies.sort_values(by='score', ascending=False, inplace=True)
        recommendations = candidate_movies.drop_duplicates().head(top_n)
        recommendations['movieId'] = recommendations.index
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations