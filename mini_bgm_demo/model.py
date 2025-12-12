import random

class FakeGenreClassifier:
    def __init__(self, id_to_genre):
        self.id_to_genre = id_to_genre
        self.genres = list(id_to_genre.values())

    def predict(self, text):
        scores = {genre: random.random() for genre in self.genres}
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        best_genre = top3[0][0]
        return best_genre, top3
