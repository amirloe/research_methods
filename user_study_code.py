import numpy as np
import pandas as pd
from explicitMF import ExplicitMF


def load_dataset():
    df = pd.read_csv('datamatrix.csv')
    ratings = df.fillna(0)
    return ratings


def get_train_test_artist_id():
    user_study_df = pd.read_csv('user_study_data.csv')
    artists_ids_train = user_study_df.loc[user_study_df['is_conceald'] == 0]['artist_id']
    artists_ids_test = user_study_df.loc[user_study_df['is_conceald'] == 1]['artist_id']
    return artists_ids_train, artists_ids_test


def train_test_split(ratings):
    ratings_np = ratings.drop(['user_id'], axis=1).to_numpy()
    test = np.zeros(ratings_np.shape)
    train = ratings_np.copy()
    concealed = []

    artists_ids_train, artists_ids_test = get_train_test_artist_id()
    for index, row in ratings.iterrows():
        if row['user_id'] == 2500:
            test[index, :] = 0.
            train[index, :] = 0.
            col = ratings.columns[1:]
            train_indices = pd.Series(col)[pd.Series(col).isin(artists_ids_train.apply(str))].index.to_numpy()
            test_indices = pd.Series(col)[pd.Series(col).isin(artists_ids_test.apply(str))].index.to_numpy()
            train[index, train_indices] = ratings_np[index, train_indices]
            test[index, test_indices] = ratings_np[index, test_indices]
        else:
            test_ratings = np.random.choice(ratings_np[index, :].nonzero()[0],
                                            size=300,
                                            replace=False)

            train[index, test_ratings] = 0.
            test[index, test_ratings] = ratings_np[index, test_ratings]
            concealed.append(test_ratings)
    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    # remember concealed
    return train, test, np.array(concealed)


dataset = load_dataset()
rating_np = dataset.to_numpy()
train, test, concealed_idx = train_test_split(dataset)
mf_mode = ExplicitMF(rating_np,n_factors=40,user_reg=.01,item_reg=.01, concealed=concealed_idx, verbose=True)
mf_mode.train(50)
_, artists_ids_test = get_train_test_artist_id()

col = dataset.columns[1:]
test_indices = pd.Series(col)[pd.Series(col).isin(artists_ids_test.apply(str))].index.to_numpy()

user_to_predict = dataset[dataset['user_id'] == 2500].index[0]

predictions = []

for item in test_indices:
    predictions.append(mf_mode.predict(user_to_predict,item))
test_predicition = test[user_to_predict,test_indices]
test_top_3 = np.array(test_predicition).argsort()[::-1][:3]
pred_top_3 = np.array(predictions).argsort()[::-1][:3]
print(f'Real top 3 = {test_top_3}  \n Model top 3 = {pred_top_3}')
# for item in items:
#     mf_mode.predict(2500,item)
