from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import numpy as np


class KnnModel():
    def __init__(self, metric='cosine', n_neighbors=5, topN=10):
        self.model_knn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors + 1)
        self.topN = topN
        self.num_of_users = -1
        self.num_of_artists = -1
        self.train = -1
        self.test = -1
        self.concealed = -1

    def run_model(self, train, test, concealed):
        self.num_of_users, self.num_of_artists = train.shape
        self.train, self.test, self.concealed = train, test, concealed
        self.model_knn.fit(train)
        # number of neighbors is the number of recommending items
        real_ratings = []
        for user in range(self.num_of_users):
            real_ratings.append(test[user,self.concealed[user]])
        real_ratings = np.array(real_ratings)
        distances, indices = self.model_knn.kneighbors(train)
        predicted_values = self._get_predictions(train, distances, indices)
        precision, recall = self._clf_analysis(real_ratings, predicted_values)
        return precision, recall

    def _get_predictions(self, train, distances, indices):
        users_predictions = []
        for user_idx in range(self.num_of_users):
            sim_users_ratings = np.array([train[i] for i in indices[user_idx]][1:])
            '''
            print(f'this is the users indices {indices[idx]}')
            print(f'ratings of user{indices[idx][1]} are:\n{train[indices[idx][1]].shape}')
            '''
            users_predictions.append(self._get_prediction(user_idx, np.reshape(distances[user_idx][1:], (5, 1)), sim_users_ratings))
        print(np.shape(users_predictions))
        return np.array(users_predictions).reshape((self.num_of_users,self.topN))

    def _get_prediction(self,user_idx, distances, ratings):
        # return predictions as n sized array of indices of items
        weights = np.subtract(np.ones((5, 1)), distances)
        total_weight = sum(weights)
        weights = weights / total_weight
        predicted_ratings = np.zeros((self.num_of_artists, 1))
        idx = 0
        for rating in ratings:
            predicted_ratings += weights[idx] * np.reshape(rating, (self.num_of_artists, 1))
            idx += 1
        return self._best_not_rated(user_idx, predicted_ratings)

    def _best_not_rated(self, user_idx, predicted_ratings):
        idx_row = np.argsort(predicted_ratings[self.concealed[user_idx]], axis=0)
        item_indices = idx_row[::-1][:self.topN]
        return item_indices

    def _clf_analysis(self, real, pred):
        precision = 0
        recall = 0
        total_tp = 0
        for user in range(self.num_of_users):
            real_top = real[user].argsort()[::-1][:self.topN*2]
            pred_top = pred[user]
            tp = len(np.intersect1d(real_top, pred_top))
            precision += len(pred_top)
            recall += len(real_top)
            total_tp += tp
        return total_tp / precision , total_tp/recall

