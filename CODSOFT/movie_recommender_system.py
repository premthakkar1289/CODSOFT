import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'userId': [1, 1, 1, 2, 2],
    'movieId': [1, 3, 6, 1, 3],
    'rating': [4.0, 4.0, 4.0, 3.0, 5.0],
    'timestamp': [964982703, 964981247, 964982224, 835355493, 835355499]
}
ratings_df = pd.DataFrame(data)
ratings_df.to_csv('ratings.csv', index=False)
data = {
    'movieId': [1, 2, 3,4,5,6],
    'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men','bhulbhulaiya','hera pheri','american pysho'],
    'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Comedy|Romance','Comedy|Horror','Comedy|Romance','Comedy|Horror']
}
movies_df = pd.DataFrame(data)
movies_df.to_csv('movies.csv', index=False)

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
movie_ratings = pd.merge(ratings, movies, on='movieId')
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix = user_movie_matrix.fillna(0)
movie_similarity = cosine_similarity(user_movie_matrix.T)
def get_movie_recommendations(movie_title, top_n=5):
    movie_index = user_movie_matrix.columns.get_loc(movie_title)
    movie_sim_scores = movie_similarity[movie_index]
    similar_movies_indices = movie_sim_scores.argsort()[::-1][1:top_n+1]
    similar_movies = user_movie_matrix.columns[similar_movies_indices]
    return similar_movies
movie_title = 'american pysho'
recommendations = get_movie_recommendations(movie_title)
print("Top 5 recommendations for movie", movie_title, ":\n", recommendations)
