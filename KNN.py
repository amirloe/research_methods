from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

def knn_model(metric='cosine', n_neighbors=3):
    model_knn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

