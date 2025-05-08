import numpy as np
import pandas as pd
import ast
import json

from irt import *
from utils import *


# Load IRT parameters
#A, B, _ = load_irt_parameters('data/mmlu_pro_all_irt_model/')

test_data = pd.read_json('./data/test_mmlu_with_ids.json')
n_test_samples = len(test_data)
df_alpha = pd.read_csv('knn_irt_predicted_alpha_k_20.csv')
df_difficulty = pd.read_csv('knn_irt_predicted_difficulty_k_20.csv')

A, B, Theta = load_irt_parameters('./data/mmlu_data_training_irt_model/')

# Parse the predicted_alpha column into a list of lists
alpha_list = df_alpha['predicted_alpha'].apply(ast.literal_eval).tolist()
alpha_array = np.array(alpha_list).T  
 
difficulty_array = np.array(df_difficulty['predicted_difficulty'].astype(float)).reshape(1, n_test_samples)

with open('./data/mmlu_data_training_irt_model/best_parameters.json', 'r') as f:
    best_params = json.load(f)

# Prepare new discrimination values as lists (each entry is a 10-dimensional array)
test_discs = [list(alpha_array[:, i]) for i in range(alpha_array.shape[1])]

# Prepare new difficulty values as a flat list
test_diffs = [[val] for val in difficulty_array.flatten()]
if 'diff' in best_params:
    best_params['diff'].extend(test_diffs)
else:
    raise ValueError("The 'diff' key is missing or not long enough in best_parameters.json.")

# Append the new discrimination values to 'disc'
if 'disc' in best_params:
    best_params['disc'].extend(test_discs)
else:
    raise ValueError("The 'disc' key is missing in best_parameters.json.")

print('final diff length',len(best_params['diff']))
print('final disc length',len(best_params['disc']))

# Save to a new JSON file
with open('./data/mmlu_data_training_irt_model/best_parameters.json', 'w') as f:
    json.dump(best_params, f)

