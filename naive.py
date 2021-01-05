import numpy as np


class Naive_model():
    def __init__(self, topN=10):
        self.train = -1
        self.test = -1
        self.concealed = -1
        self.sorted_idx_by_rate = -1
        self.topN = topN

    def run_model(self, train, test, concealed):
        self.train, self.test, self.concealed = train, test, concealed
        self._get_sorted_avg()
        predictions = self._get_predictions()
        # toDo: compare top N predictions for each user with real top 10 of him.

    def _get_sorted_avg(self):
        artists_avg = np.mean(self.train, axis=0)
        self.sorted_idx_by_rate = artists_avg.argsort()[::-1]

    def _get_predictions(self):
        top_predictions = []
        for user_idx in range(self.train.shape[0]):
            user_conscealed = self.concealed[user_idx]
            top_predictions.append(self._predict_for_user(user_conscealed))
        return np.array(top_predictions)

    def _predict_for_user(self, user_conscealed):
        concealed_rating_idx = np.zeros(user_conscealed.shape)
        concealed_idx = 0
        for item_idx in user_conscealed:
            curr_rating_idx = np.where(self.sorted_idx_by_rate == item_idx)[0][0]
            concealed_rating_idx[concealed_idx] = curr_rating_idx
            concealed_idx += 1
        positions = concealed_rating_idx.argsort()
        user_conscealed = user_conscealed[positions]
        return user_conscealed[:self.topN]
