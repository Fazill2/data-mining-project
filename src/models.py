from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from surprise  import SVD, KNNBasic, Reader, Dataset
from src.user_profile import UserProfileCreator
import numpy as np
from src.models import AprioriKmeansRecommender, AprioriSimplestRecommender, SimpleRegressionRecommender, SVDRecommender, GBKmeansRecommender, KMeansRecommender

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

class SimpleRegressionRecommender:
    def __init__(self, item_data: pd.DataFrame, random_state=42):
        self.item_data = item_data
        self._train_ratings = None
        self._train_profiles = None
        self._train_profiles_clusters = None
        self._centroids = None
        self._cluster_rules = {}
        self._kmeans = None

    def recommend_for_user(self, user_ratings: pd.DataFrame, evaluation_ratings: pd.DataFrame = None, top_n=5, penalize_genres=False):
        merged = user_ratings.merge(self.item_data, on='movieId')
        genre_columns = [col for col in self.item_data.columns if col not in ['movieId', 'userId', 'rating']]
        X = merged[genre_columns].values
        y = merged['rating'].values

        # Use Ridge regression for regularization
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        already_rated = set(user_ratings['movieId'])
        candidate_movies = self.item_data.copy()[genre_columns]
        candidate_movies = candidate_movies[~candidate_movies.index.isin(already_rated)]
        if evaluation_ratings is not None:
            candidate_movies = candidate_movies[candidate_movies.index.isin(evaluation_ratings['movieId'])]
        predicted_ratings = model.predict(candidate_movies)
        candidate_movies['score'] = predicted_ratings
        candidate_movies.sort_values(by='score', ascending=False, inplace=True)
        recommendations = candidate_movies.drop_duplicates().head(top_n)
        recommendations['movieId'] = recommendations.index
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations

class GBKmeansRecommender:
    def __init__(self, item_data: pd.DataFrame,  k=15, random_state=42):
        self.item_data = item_data
        self._train_ratings = None
        self.k = k
        self.random_state = random_state
        self._train_profiles = None
        self._train_profiles_clusters = None
        self._centroids = None
        self._cluster_models = {}
        self._kmeans = None

    def fit(self, train_ratings: pd.DataFrame):
        self._train_ratings = train_ratings
        self._train_profiles = UserProfileCreator.build_user_profiles(self.item_data, self._train_ratings)


        self._kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
        self._kmeans.fit(self._train_profiles)
        self._train_profiles['cluster'] = self._kmeans.labels_

        self._train_profiles_clusters = self._train_profiles.groupby('cluster')

        for cluster_id in range(self.k):
            self._train_model_for_cluster(cluster_id)

    def _train_model_for_cluster(self, cluster_id):
        cluster_user_ids = self._train_profiles[self._train_profiles['cluster'] == cluster_id].index
        cluster_ratings = self._train_ratings[self._train_ratings['userId'].isin(cluster_user_ids)]

        merged = cluster_ratings.merge(self.item_data, on='movieId')

        genre_columns = [col for col in self.item_data.columns if col not in ['movieId', 'userId', 'rating']]
        X = merged[genre_columns].values
        y = merged['rating'].values
        model = GradientBoostingRegressor(random_state=self.random_state)
        model.fit(X, y)

        self._cluster_models[cluster_id] = model

    def recommend_for_user(self, user_ratings: pd.DataFrame, evaluation_ratings: pd.DataFrame = None, top_n=5):
        user_profile = UserProfileCreator.build_user_profiles(self.item_data, user_ratings)
        user_cluster = self._kmeans.predict(user_profile.values.reshape(1, -1))[0]

        model = self._cluster_models.get(user_cluster)
        already_rated = set(user_ratings['movieId'])
        genre_columns = [col for col in self.item_data.columns if col not in ['movieId', 'userId', 'rating']]
        candidate_movies = self.item_data.copy()[genre_columns]
        candidate_movies = candidate_movies[~candidate_movies.index.isin(already_rated)]
        if evaluation_ratings is not None:
            candidate_movies = candidate_movies[candidate_movies.index.isin(evaluation_ratings['movieId'])]
        predicted_ratings = model.predict(candidate_movies)
        candidate_movies['score'] = predicted_ratings
        candidate_movies.sort_values(by='score', ascending=False, inplace=True)
        recommendations = candidate_movies.drop_duplicates().head(top_n)
        recommendations['movieId'] = recommendations.index
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations

class SVDRecommender:
    def __init__(self, item_data: pd.DataFrame, random_state=42, n_epochs=20, n_factors=100, lr_all=0.005, reg_all=0.02):
        self.random_state = random_state
        self.reader = Reader(rating_scale=(0.5, 5.0))
        self.model = SVD(n_epochs=n_epochs, n_factors=n_factors, lr_all=lr_all, reg_all=reg_all, random_state=self.random_state)
        self._train_ratings = None
        genre_columns = [col for col in item_data.columns if col not in ['movieId', 'userId', 'rating']]
        self.candidate_movies = item_data.copy()
        self.candidate_movies['movieId'] = self.candidate_movies.index
        self.candidate_movies.reset_index(drop=True, inplace=True)
        self.candidate_movies = self.candidate_movies.drop(columns=genre_columns)

    def fit(self, ratings: pd.DataFrame):
        self._train_ratings = ratings
        train_data = Dataset.load_from_df(self._train_ratings, self.reader)
        trainset = train_data.build_full_trainset()
        self.model.fit(trainset)

    def recommend_for_user(self, user_ratings: pd.DataFrame, user_id, evaluation_ratings: pd.DataFrame = None, top_n=5):
        already_rated = set(user_ratings['movieId'])
        candidate_movies = self.candidate_movies.copy()
        candidate_movies = candidate_movies[~candidate_movies.index.isin(already_rated)]
        if evaluation_ratings is not None:
            candidate_movies = candidate_movies[candidate_movies['movieId'].isin(evaluation_ratings['movieId'])]
        scores = []
        for _, row in candidate_movies.iterrows():
            prediction = self.model.predict(uid=user_id, iid=row['movieId'])
            score = prediction.est
            scores.append(score)
        candidate_movies['score'] = scores
        candidate_movies.sort_values(by='score', ascending=False, inplace=True)
        recommendations = candidate_movies.drop_duplicates().head(top_n)
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations

class KMeansRecommender:
    def __init__(self, item_data: pd.DataFrame, k=15, random_state=42):
        self.item_data = item_data
        self.k = k
        self.random_state = random_state
        self.kmeans = None
        self.clusters = None
        
    def fit(self, train_ratings: pd.DataFrame):
        # Create and train k-means model
        genre_columns = [col for col in self.item_data.columns if col.startswith('genres_')]
        X = self.item_data[genre_columns].values
        
        self.kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
        self.kmeans.fit(X)
        # Assign each movie to its corresponding cluster
        self.clusters = pd.Series(self.kmeans.labels_, index=self.item_data.index)
        
        return self
    
    def recommend_for_user(self, user_ratings: pd.DataFrame, evaluation_ratings: pd.DataFrame = None, top_n=5):
        if self.kmeans is None:
            raise ValueError("Model has not been trained yet. Call fit() method first.")
        
        user_movie_ids = user_ratings['movieId'].values
        user_clusters = set()
        for movie_id in user_movie_ids:
            if movie_id in self.clusters.index:
                user_clusters.add(self.clusters[movie_id])
        
        candidate_movies = self.item_data.copy()
        if evaluation_ratings is not None:
            candidate_movies = candidate_movies[candidate_movies.index.isin(evaluation_ratings['movieId'])]
        
        candidate_movies = candidate_movies[~candidate_movies.index.isin(user_movie_ids)]
        scores = {}
        for movie_id in candidate_movies.index:
            movie_cluster = self.clusters[movie_id]
            # Movies from the same clusters as already rated get 0.5 points
            if movie_cluster in user_clusters:
                scores[movie_id] = 0.5
            else:
                scores[movie_id] = 0.0
        
        recommendations = pd.DataFrame({
            'movieId': list(scores.keys()),
            'prediction': list(scores.values())
        })
        
        recommendations = recommendations.sort_values('prediction', ascending=False)
        
        return recommendations.head(top_n)