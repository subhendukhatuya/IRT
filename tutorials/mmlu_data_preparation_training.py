import pandas as pd
import numpy as np
import pickle

def process_data(train_file, output_file, num_models):
    # Read the CSV files
    train_df = pd.read_json(train_file)
 
    # Create a mapping of original question IDs to sequential IDs
    unique_questions = train_df['question_id'].unique()
    question_id_map = {qid: idx for idx, qid in enumerate(unique_questions)}
    
    # Map the question IDs
    train_df['sequential_id'] = train_df['question_id'].map(question_id_map)
    
    # Initialize correctness array with shape (num_samples, num_models)
    correctness = np.zeros((len(train_df), num_models), dtype=float)
    
    # Process each question and set correctness based on target for first set
    for _, row in train_df.iterrows():
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


    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data processed and saved to {output_file}")


if __name__ == "__main__":
    # Define file paths
    train_file = "./data/train_mmlu_with_ids.json"
    num_models = 7
    output_file = "./data/mmlu_data_training.pkl"
  
    # Process the data
    process_data(train_file, output_file, num_models)


