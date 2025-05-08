import numpy as np
import pickle
from tqdm import tqdm
from irt import *
from utils import *
import pandas as pd

random_state = 42

with open('./data/mmlu_data_all.pkl', 'rb') as handle:
    data = pickle.load(handle)

scenarios_position, subscenarios_position = prepare_data(scenarios, data)
Y = create_responses(scenarios, data)
print('Y.shape',Y.shape)

balance_weights = np.ones(Y.shape[1])

Y_test = Y

with open('./data/mmlu_data_training_anchor.pickle', 'rb') as handle:
    anchor = pickle.load(handle)

anchor_points = anchor['anchor_points']
anchor_weights = anchor['anchor_weights']

print('anchor_points',anchor_points['mmlu'])
print('anchor_weights',anchor_weights['mmlu'])


A, B, _ = load_irt_parameters('./data/mmlu_data_training_irt_model/')
seen_items = np.hstack([np.array(scenarios_position[scenario])[anchor_points[scenario]] for scenario in scenarios.keys()]).tolist()
print('seen items shape',np.array(seen_items).shape)
unseen_items = [i for i in range(Y_test.shape[1]) ]
print('unseen_items shape',np.array(unseen_items).shape)



thetas = [estimate_ability_parameters(Y_test[j][seen_items], A[:, :, seen_items], B[:, :, seen_items]) for j in tqdm(range(Y_test.shape[0]))]


pirt_preds = {}

for scenario in scenarios.keys():

    ind_seen = [u for u in seen_items if u in scenarios_position[scenario]]
    ind_unseen = [u for u in unseen_items if u in scenarios_position[scenario]]
    pirt_lambd = len(anchor_points['mmlu'])/len(scenarios_position[scenario])

    pirt_pred = []
    pirt_preds[scenario] = {}  # Initialize dictionary for this scenario

    for j in range(Y_test.shape[0]):
        data_part = (balance_weights*Y_test)[j,ind_unseen]
        irt_part = (balance_weights*item_curve(thetas[j], A, B))[0,ind_unseen]
        pirt_pred = pirt_lambd*data_part + (1-pirt_lambd)*irt_part
        pirt_preds[scenario][j] = pirt_pred  # Store pirt_pred_2 for this j

# Save pirt_preds_2 to a pickle file
with open('./data/mmlu_data_pirt_preds.pkl', 'wb') as handle:
    pickle.dump(pirt_preds, handle)
print("Saved pirt_preds_2 to ./data/mmlu_data_pirt_preds.pkl")


