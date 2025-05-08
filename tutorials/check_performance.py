import torch
import json
import pandas as pd
import random
import ast

def calculate_acc_predict(csv_file):
    # Read the CSV file
    test_data = pd.read_json('./data/test_mmlu_with_ids.json')
    df = pd.read_csv(csv_file)
    df['scores'] = test_data['scores']
    correct_predict = 0
    num_samples = len(df)

    for _, row in df.iterrows():
        scores_dict = dict(row['scores'])
        best_model = row['Pred']
        scores_tensor = torch.tensor([scores_dict[m] for m in scores_dict])
        model_names = list(scores_dict.keys())
        best_model_index = model_names.index(best_model)
        # Create mask for the selected model
        mask = torch.zeros_like(scores_tensor)
        mask[best_model_index] = 1
        # Binarize scores
        scores_tensor[scores_tensor > 0] = 1
        correct_predict += (scores_tensor * mask).sum().item()
    
    acc_predict = correct_predict / num_samples if num_samples > 0 else 0
    print("MMLU acc_predict:", acc_predict)
    return acc_predict

# Example usage
if __name__ == "__main__":
    calculate_acc_predict('p_irt_predictions.csv')

