import pickle

def get_genres():
    return ["Action", "Adventure", "Animation",
            "Children", "Comedy", "Crime", "Documentary",
            "Drama", "Fantasy", "Film-Noir", "Horror",
            "IMAX", "Musical", "Mystery", "Romance",
            "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)"]

def save_model(model, filename):
    with open(filename, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
