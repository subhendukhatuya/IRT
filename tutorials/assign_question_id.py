import pandas as pd

# Load the JSON file (assuming it's a list of dicts)
df_train = pd.read_json("./data/mmlu_train.json")

# Assign sequential question_id for train
df_train['question_id'] = range(len(df_train))

# Save back to JSON if needed
df_train.to_json("./data/train_mmlu_with_ids.json", orient='records', lines=False)

# Now process test.json, starting IDs after train
df_test = pd.read_json("./data/mmlu_test.json")
start_id = df_train['question_id'].max() + 1 if len(df_train) > 0 else 0
df_test['question_id'] = range(start_id, start_id + len(df_test))
df_test.to_json("./data/test_mmlu_with_ids.json", orient='records', lines=False)