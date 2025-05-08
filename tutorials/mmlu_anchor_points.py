import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from irt import *
from utils import *
import pandas as pd

random_state = 42


with open('./data/mmlu_data_training.pkl', 'rb') as handle:
    data = pickle.load(handle)

scenarios_position, subscenarios_position = prepare_data(scenarios, data)
Y = create_responses(scenarios, data)
print(Y.shape)


balance_weights = np.ones(Y.shape[1])

Y_train = Y

number_item = 100
clustering = 'irt' # 'correct.' or 'irt'

anchor_points = {}
anchor_weights = {}
for scenario in scenarios.keys():
    if clustering=='correct.':
        X = Y_train[:,scenarios_position[scenario]].T
    elif clustering=='irt':
        A, B, _ = load_irt_parameters('./data/mmlu_data_training_irt_model/')
        X = np.vstack((A.squeeze(), B.squeeze().reshape((1,-1)))).T
        X = X[scenarios_position[scenario]]
    else:
        raise NotImplementedError 
        
    #Normalizing balance_weights, so their sum is one within each scenario
    norm_balance_weights = balance_weights[scenarios_position[scenario]]
    norm_balance_weights /= norm_balance_weights.sum()

    # Fitting the KMeans model
    kmeans = KMeans(n_clusters=number_item, n_init="auto", random_state=random_state)
    kmeans.fit(X, sample_weight=norm_balance_weights)

    # Calculating anchor points
    anchor_points[scenario] = pairwise_distances(kmeans.cluster_centers_, X, metric='euclidean').argmin(axis=1)

    # Calculating anchor weights
    anchor_weights[scenario] = np.array([np.sum(norm_balance_weights[kmeans.labels_==c]) for c in range(number_item)])


anchor = {'anchor_points':anchor_points,
          'anchor_weights':anchor_weights}

with open('./data/mmlu_data_training_anchor.pickle', 'wb') as handle:
    pickle.dump(anchor, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(anchor_points['mmlu'])
print(anchor_weights['mmlu'])