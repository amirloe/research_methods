from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error


class KnnModel():
    def __init__(self, metric='cosine', n_neighbors=3):
        self.model_knn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)

    def run_model(self, dataSet):
        self.model_knn.fit(dataSet)

        get_recommendations(dataSet)


def get_recommendations(user_vector):
    pass
