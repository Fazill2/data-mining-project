import pandas as pd

class RankingCreator:
    @staticmethod
    def create_ranking(user_profile: pd.Series, movie_rec_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a ranking of movies based on the user profile and movie recommendations.

        Parameters:
        user_profile (pd.Series): The user profile containing genre preferences and ratings.
        movie_rec_df (pd.DataFrame): The movie recommendation DataFrame.

        Returns:
        pd.DataFrame: The ranked movies with their scores.
        """
        ranking: pd.DataFrame = movie_rec_df.copy()
        ranking['score'] = 0

        for row in user_profile.index:
            ranking['score'] += ranking[row] * user_profile[row]

        ranking = ranking.sort_values(by='score', ascending=False)

        return ranking