import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def load_pirt_predictions(file_path):

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def process_predictions(predictions, test_data):

    # Model names as provided
    model_names = [
        "mistralai/Mistral-7B-v0.1",
        "meta-math/MetaMath-Mistral-7B",
        "itpossible/Chinese-Mistral-7B-v0.1",
        "HuggingFaceH4/zephyr-7b-beta",
        "cognitivecomputations/dolphin-2.6-mistral-7b",
        "meta-llama/Meta-Llama-3-8B",
        "cognitivecomputations/dolphin-2.9-llama3-8b"
    ]

    test_samples_length = len(test_data)
    # Extract predictions for each model
    preds = [predictions['mmlu'][i][-test_samples_length:] for i in range(7)]
    # Build DataFrame
    results = pd.DataFrame({'question_id': test_data['question_id']})
    for col, pred in zip(model_names, preds):
        results[col] = pred
    # Find the index of the max prediction for each row
    pred_array = results[model_names].values
    max_indices = np.argmax(pred_array, axis=1)
    # Assign Pred as the model name with the highest score
    results['Pred'] = [model_names[idx] for idx in max_indices]
    # Return all columns
    return results[['question_id'] + model_names + ['Pred']]

def main():
    # Load predictions
    predictions = load_pirt_predictions('./data/mmlu_data_pirt_preds.pkl')
 
    
    if predictions is not None:
        # Process predictions and create final labels
        test_data = pd.read_json('./data/test_mmlu_with_ids.json')
        results = process_predictions(predictions, test_data)
        
        # Save results to CSV
        output_path = 'p_irt_predictions.csv'
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()