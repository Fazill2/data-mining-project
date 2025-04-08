import ranking
import data_loading
import data_transformation
import user_profile


if __name__ == "__main__":
    data_loader = data_loading.DataLoader('../data')
    movies_df = data_loader.load_movies()
    ratings_df = data_loader.load_ratings()
    user_ratings = ratings_df[ratings_df['userId'] == 1]
    movie_rec_df = data_transformation.DataTransformer.transform_data(movies_df, ratings_df)
    user_profile = user_profile.UserProfileCreator.build_user_profile(movie_rec_df, user_ratings)
    ranking_df = ranking.RankingCreator.create_ranking(user_profile, movie_rec_df)
    print(ranking_df.head(10))
