import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

from irt import *
from utils import *

# 1. Load data
train_df = pd.read_json('./data/train_mmlu_with_ids.json')
test_df = pd.read_json('./data/test_mmlu_with_ids.json')

# 2. Extract questions
train_questions = train_df['question'].tolist()
test_questions = test_df['question'].tolist()

# 3. Load model and encode
model = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = model.encode(train_questions, show_progress_bar=True)
test_embeddings = model.encode(test_questions, show_progress_bar=True)

# 4. Find k nearest neighbors for each test sample
k = 20  # or any value you want
similarities = cosine_similarity(test_embeddings, train_embeddings)
nearest_indices = np.argsort(-similarities, axis=1)[:, :k]  # negative for descending order

test_ids = test_df['question_id'].tolist()  
A, B, _ = load_irt_parameters('./data/mmlu_data_training_irt_model/')
B_train = B[:,:,:len(train_df)]
B_train = B_train.squeeze()

print('B_train shape',B_train.shape)

# For each test sample, get the average difficulty of its k nearest neighbors
predicted_difficulties = []
for neighbors in nearest_indices:
    neighbor_difficulties = B_train[neighbors]
    avg_difficulty = np.mean(neighbor_difficulties)
    predicted_difficulties.append(avg_difficulty)

# Create a DataFrame and save as CSV
pred_df = pd.DataFrame({
    'question_id': test_ids,
    'predicted_difficulty': predicted_difficulties
})
pred_df.to_csv('knn_irt_predicted_difficulty_k_20.csv', index=False)


A_train = A[:, :, :len(train_df)] 
A_train = A_train.squeeze(0) 

# For each test sample, get the average alpha of its k nearest neighbors
predicted_alphas = []
for neighbors in nearest_indices:
    neighbor_alphas = A_train[:, neighbors]  # shape: (10, k)
    avg_alpha = np.mean(neighbor_alphas, axis=1)  # shape: (10,)
    predicted_alphas.append(avg_alpha.tolist())

# Create a DataFrame and save as CSV
alpha_df = pd.DataFrame({
    'question_id': test_ids,
    'predicted_alpha': predicted_alphas
})
alpha_df.to_csv('knn_irt_predicted_alpha_k_20.csv', index=False)
