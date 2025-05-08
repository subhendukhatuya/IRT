import numpy as np
import pickle
from tqdm import tqdm
from irt import *
from utils import *

random_state = 42

print(scenarios)

with open('./data/mmlu_data_training.pkl', 'rb') as handle:
    data = pickle.load(handle)

print(len(data['models']) ,data['models'])

scenarios_position, subscenarios_position = prepare_data(scenarios, data)
Y = create_responses(scenarios, data)
print(Y.shape)

balance_weights = np.ones(Y.shape[1])

Y_train = Y  # 

D = 50
device = 'cpu' # Either 'cuda' or 'cpu' 
epochs = 2000  # Number of epochs for IRT model training (py-irt default is 2000)
lr = .1  # Learning rate for IRT model training (py-irt default is .1)

create_irt_dataset(Y_train, './data/mmlu_data_training_irt_dataset.jsonlines')

train_irt_model(dataset_name='./data/mmlu_data_training_irt_dataset.jsonlines',
                model_name='./data/mmlu_data_training_irt_model',
                D=D, lr=lr, epochs=epochs, device=device)  