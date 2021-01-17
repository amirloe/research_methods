import numpy as np
import pandas as pd
from explicitMF import ExplicitMF
from KNN import KnnModel
from naive import Naive_model
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind,wilcoxon


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
                                        size=300,
                                        replace=False)

        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        concealed.append(test_ratings)
    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    # remember concealed
    return train, test, np.array(concealed)


# Load data from disk
def load_dataset():
    df = pd.read_csv('datamatrix.csv', index_col=0)
    ratings = df.fillna(0)
    return ratings


def grid_search_mf(train, test,concealed_idx):
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
    best_params['precision'] = 0
    best_params['model'] = None

    for fact in latent_factors:
        print(f'Factors: {fact}')
        for reg in regularizations:
            print(f'Regularization: {reg}')
            MF_ALS = ExplicitMF(train, n_factors=fact, user_reg=reg, item_reg=reg,concealed=concealed_idx)
            MF_ALS.calculate_learning_curve(iter_array, test)
            plot_learning_curve(iter_array, MF_ALS)
            precisions_arr = [np.mean(x) for x in MF_ALS.percisions]
            max_idx = np.argmax(precisions_arr)
            if np.mean(MF_ALS.percisions[max_idx]) > best_params['precision']:
                best_params['n_factors'] = fact
                best_params['reg'] = reg
                best_params['n_iter'] = iter_array[max_idx]
                best_params['train_mse'] = MF_ALS.train_mse[max_idx]
                best_params['test_mse'] = MF_ALS.test_mse[max_idx]
                best_params['model'] = MF_ALS
                best_params['precision'] = np.mean(MF_ALS.percisions[max_idx])
                print('New optimal hyperparameters')
                print(pd.Series(best_params))

def grid_search_knn(train, test,concealed_idx):
    number_neighbors = [1, 3, 5, 10, 25]
    metrics = ['correlation', 'cosine']

    best_params = {}
    best_params['number_neighbors'] = number_neighbors[0]
    best_params['metrics'] = metrics[0]
    best_params['precision'] = 0

    for neighbors in number_neighbors:
        print(f'Neighbors: {neighbors}')
        for metric in metrics:
            print(f'Metric: {metric}')
            knn_model = KnnModel(topN=30, n_neighbors=neighbors, metric=metric)
            knn_precision, knn_recall = knn_model.run_model(train, test, concealed_idx)
            precisions_arr = [np.mean(x) for x in knn_precision]
            max_idx = np.argmax(precisions_arr)
            if np.mean(precisions_arr[max_idx]) > best_params['precision']:
                best_params['number_neighbors'] = neighbors
                best_params['metrics'] = metric
                best_params['precision'] = np.mean(precisions_arr[max_idx])
                print('New optimal hyperparameters')
                print(pd.Series(best_params))

def plt_bars(data):
    n_groups = 3
    precision_naive = data[0]
    precision_knn = data[1]
    precision_mf = data[2]

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8

    rects1 = plt.bar(index, precision_naive, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Naive')

    rects2 = plt.bar(index + bar_width, precision_knn, bar_width,
                     alpha=opacity,
                     color='g',
                     label='KNN')

    rects2 = plt.bar(index + 2 * bar_width, precision_mf, bar_width,
                     alpha=opacity,
                     color='r',
                     label='MF')

    plt.ylabel('Precision@n')

    plt.xticks(index + bar_width, ('n=10', 'n=30', 'n=50'))
    plt.legend()

    plt.tight_layout()
    plt.savefig('precision_plt.png')

# rating is a data frame with zeros instead of none
dataset = load_dataset()
rating_np = dataset.to_numpy()
artists_means = np.nanmean(dataset.replace(0, np.NaN), axis=0)
quar_vals = np.arange(0, 1, 0.1)
quartiles = np.quantile(artists_means, quar_vals)
plt.plot(quar_vals,quartiles)
plt.savefig('AVG_rating_per_quartile')
#
# #
train, test, concealed_idx = train_test_split(rating_np)
# # knn_model evaluation
# '''
# knn_model = KnnModel()
# precision,recall = knn_model.run_model(train, test, concealed_idx)
# print('Precision: ' + str(precision))
# print('Recall: ' + str(recall))
# '''
# # grid_search_mf(train, test,concealed_idx)
results = [[],[],[]]
for topN in [10,30,50]:
    print("=====KNN MODEL=====")
    # grid_search_knn(train, test,concealed_idx)

    knn_model = KnnModel(topN=topN,n_neighbors=250)
    knn_precision,knn_recall = knn_model.run_model(train, test, concealed_idx)
    knn_mean_precision = np.mean(knn_precision)
    results[1].append(knn_mean_precision)
    print('Precision: ' + str(knn_mean_precision))
    print('Recall: ' + str(np.mean(knn_recall)))
    # matrix factorization evaluation

    '''
    Matrix factorization
    grid_search_mf(train, test)
    '''


    print("=====MF MODEL=====")
    mf_mode = ExplicitMF(train, n_factors=20, user_reg=100, item_reg=100,concealed=concealed_idx,topN=topN)
    mf_mode.calculate_learning_curve([50],test)
    mf_precision, mf_recall = mf_mode.get_percisions_recalls()
    mf_mean_precision = np.mean(mf_precision)
    results[2].append(mf_mean_precision)
    print('Precision: ' + str(mf_mean_precision))
    print('Recall: ' + str(np.mean(mf_recall)))


    # naive model evaluation
    print("=====Naive MODEL=====")
    naive_model = Naive_model(topN=topN)
    naive_precision, naive_recall = naive_model.run_model(train, test, concealed_idx)
    naive_mean_precision = np.mean(naive_precision)
    results[0].append(naive_mean_precision)
    print('Precision: ' + str(naive_mean_precision))
    print('Recall: ' + str(np.mean(naive_recall)))

    print(ttest_ind(knn_precision, mf_precision))
    print(wilcoxon(knn_precision,mf_precision))

    print(ttest_ind(naive_precision, mf_precision))
    print(wilcoxon(naive_precision,mf_precision))

plt_bars(results)