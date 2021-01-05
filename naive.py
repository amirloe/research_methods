

class Naive_model():
    def __init__(self,ratings, concealed):
        self.ratings = ratings.to_numpy()
        self.concealed = concealed

    def predict_for_user(self,user):



    def get_top_n(self, n, prediction, user):

        return prediction[user, self.concealed[user]].argsort()[::-1][:n]
