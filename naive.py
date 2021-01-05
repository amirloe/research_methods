import numpy as np

class Naive_model():
    def __init__(self,train,test, concealed):
        self.train = train.to_numpy()
        self.test = test
        self.concealed = concealed

    def run_model(self):
        sorted_avg = self._get_sorted_avg()

    def _get_sorted_avg(self):
        artists_avg = np.mean(self.train,axis=1)
        top_10 = artists_avg.argsort()[::-1]

    def predict_for_user(self,user):
        pass

