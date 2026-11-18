import numpy as np
from sklearn.neighbors import NearestNeighbors

movies = [
    "Toy Story", "Finding Nemo", "Monsters, Inc.", "The Incredibles",
    "Shrek", "Jurassic Park", "Indiana Jones", "The Hangover",
    "The Shawshank Redemption", "The Pursuit of Happyness",
]

features = np.array([
    [1, 1, 1, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
], dtype=float)

liked_movies = ["Toy Story", "Monsters, Inc.", "Finding Nemo"]
hidden_favorite = "Finding Nemo"
other_favorites = [movie for movie in liked_movies if movie != hidden_favorite]

model = NearestNeighbors(metric="euclidean")
model.fit(features)

def recommend_movies(favorites, number=3):
    favorite_positions = [movies.index(movie) for movie in favorites]
    viewer_profile = features[favorite_positions].mean(axis=0).reshape(1, -1)
    distances, positions = model.kneighbors(viewer_profile, n_neighbors=len(movies))
    ranked = sorted(zip(distances[0], positions[0]), key=lambda pair: (pair[0], pair[1]))
    return [movies[position] for _, position in ranked
            if movies[position] not in favorites][:number]

recommendations = recommend_movies(other_favorites)
print("Favorites used:", ", ".join(other_favorites))
print("\nRecommended movies:")
for rank, title in enumerate(recommendations, start=1):
    print(f"{rank}. {title}")

passed = hidden_favorite in recommendations
print("\nHidden favorite:", hidden_favorite)
print("Did it appear in our top 3?", passed)
print("This is one small test with hand-made features, not proof of real-world quality.")
