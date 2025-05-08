import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def process_data(train_file, test_file, output_file, num_models):
    # Read the CSV files
    train_df = pd.read_json(train_file)
    test_df = pd.read_json(test_file)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Create a mapping of original question IDs to sequential IDs
    unique_questions = combined_df['question_id'].unique()
    question_id_map = {qid: idx for idx, qid in enumerate(unique_questions)}
    
    # Map the question IDs
    combined_df['sequential_id'] = combined_df['question_id'].map(question_id_map)
    
    # Initialize correctness array with shape (num_samples, num_models)
    correctness = np.zeros((len(combined_df), num_models), dtype=float)
    
    # Process each question and set correctness based on target for first set
    for _, row in combined_df.iterrows():
        qid = row['sequential_id']
        scores = row['scores']

        # Assign correctness based on scores
        model_idx = 0
        for model_score in scores.values():
            correctness[qid, model_idx] = 1 if model_score > 0 else 0
            model_idx += 1

    
    # Create the final data structure
    data = {
        'data': {
            'mmlu': {
                'correctness': correctness
            }
        },
        'models': ["mistralai\/Mistral-7B-v0.1","meta-math\/MetaMath-Mistral-7B","itpossible\/Chinese-Mistral-7B-v0.1","HuggingFaceH4\/zephyr-7b-beta","cognitivecomputations\/dolphin-2.6-mistral-7b","meta-llama\/Meta-Llama-3-8B","cognitivecomputations\/dolphin-2.9-llama3-8b"]
    }
    # Extract model names from scores dictionary

    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data processed and saved to {output_file}")

if __name__ == "__main__":
    # Define file paths
    train_file = "./data/train_mmlu_with_ids.json"
    test_file = "./data/test_mmlu_with_ids.json"
    num_models = 7
    output_file = "./data/mmlu_data_all.pkl"
    # Process the data
    process_data(train_file, test_file, output_file, num_models)

