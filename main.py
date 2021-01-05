import numpy as np
import pandas as pd
from explicitMF import ExplicitMF
from KNN import KnnModel
from matplotlib import pyplot as plt

np.random.seed(0)


def plot_learning_curve(iter_array, model):
    plt.plot(iter_array, model.train_mse, label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse, label='Test', linewidth=5)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('iterations', fontsize=30)
    plt.ylabel('MSE', fontsize=30)
    plt.legend(loc='best', fontsize=20)
    plt.savefig(f'fig_on_{model.n_factors}_ureg{model.user_reg}_ireg{model.item_reg}.png')
    plt.clf()


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    concealed = []
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=50,
                                        replace=False)

        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        concealed.append(test_ratings)
    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    # remember concealed
    return train, test, concealed


# Load data from disk
def load_dataset():
    df = pd.read_csv('datamatrix.csv', index_col=0)
    ratings = df.fillna(0)
    return ratings


def grid_search_mf(train, test):
    latent_factors = [5, 10, 20, 40, 80]
    regularizations = [0.01, 0.1, 1., 10., 100.]
    regularizations.sort()
    iter_array = [1, 2, 5, 10, 25, 50, 100]

    best_params = {}
    best_params['n_factors'] = latent_factors[0]
    best_params['reg'] = regularizations[0]
    best_params['n_iter'] = 0
    best_params['train_mse'] = np.inf
    best_params['test_mse'] = np.inf
    best_params['model'] = None

    for fact in latent_factors:
        print(f'Factors: {fact}')
        for reg in regularizations:
            print(f'Regularization: {reg}')
            MF_ALS = ExplicitMF(train, n_factors=fact, user_reg=reg, item_reg=reg)
            MF_ALS.calculate_learning_curve(iter_array, test)
            plot_learning_curve(iter_array, MF_ALS)
            min_idx = np.argmin(MF_ALS.test_mse)
            if MF_ALS.test_mse[min_idx] < best_params['test_mse']:
                best_params['n_factors'] = fact
                best_params['reg'] = reg
                best_params['n_iter'] = iter_array[min_idx]
                best_params['train_mse'] = MF_ALS.train_mse[min_idx]
                best_params['test_mse'] = MF_ALS.test_mse[min_idx]
                best_params['model'] = MF_ALS
                print('New optimal hyperparameters')
                print(pd.Series(best_params))


# rating is a data frame with zeros instead of none
dataset = load_dataset()
rating_np = dataset.to_numpy()
artists_means = np.nanmean(dataset.replace(0, np.NaN), axis=0)
quar_vals = np.arange(0.1, 1, 0.1)
quartiles = np.quantile(artists_means, quar_vals)
plt.plot(quar_vals, quartiles)
# plt.show()

#
train, test, concealed_idx = train_test_split(rating_np)
# knn_model evaluation
knn_model = KnnModel()
knn_model.run_model(train, test, concealed_idx)

# mf_mode = ExplicitMF(rating_np,concealed=concealed_idx,verbose=True)
# mf_mode.calculate_learning_curve([1],test)
# Matrix factorization
# grid_search_mf(train, test)
