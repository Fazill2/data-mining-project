import pandas as pd

class DataLoader:
    """
    A class to load and preprocess data for machine learning tasks.
    """

    def __init__(self, data_dir: str):
        """
        Initializes the DataLoader with the path to the data file.
        
        Parameters:
        data_path (str): The path to the data file.
        """
        self.data_path = data_dir

    def load_movies(self) -> pd.DataFrame:
        """
        Loads the movies dataset from the specified path.

        Returns:
        pd.DataFrame: The loaded movies dataset.
        """
        movies_df = pd.read_csv(self.data_path + '/movies.csv')
        return movies_df

    def load_ratings(self) -> pd.DataFrame:
        """
        Loads the ratings dataset from the specified path.

        Returns:
        pd.DataFrame: The loaded ratings dataset.
        """
        ratings_df = pd.read_csv(self.data_path + '/ratings.csv')
        return ratings_df