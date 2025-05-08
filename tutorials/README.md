### Prerequisites

Ensure that you have Python 3.11.7 installed on your machine. You can download it from the [official Python website](https://www.python.org/downloads/).

After ensuring you have the correct Python version, install all the required packages listed in `requirements.txt` using `pip`. Run the following command:

```shell
pip install -r requirements.txt
```

```shell
git clone https://github.com/felipemaiapolo/py-irt.git
cd py-irt
poetry install
```

## Tutorials Overview

This repository contains three Jupyter notebooks. The following notebooks will by default run on tinybenchmark datasets and models. If you want to skip and directly move to run for MMLU then jump to MMLU Data Preparation. If you want to tun the following noteboks, delete line no 147 in `utils.py`. We just overwriting scenarios with mmlu by adding `scenarios = {'mmlu':['mmlu']}` in line 147 of `utils.py`.

1. **Training IRT Models (`training_irt.ipynb`):**
   - This notebook demonstrates how to train your own Item Response Theory (IRT) models. It covers the setup, training process, and evaluation of the models.

2. **Finding Anchor Points (`anchor_points.ipynb`):**
   - In this tutorial, we show how to identify anchor points from your training set. These anchor points are crucial for estimating the performance of new models on the test set.

3. **Estimating LLM Performance (`estimating_performance.ipynb`):**
   - This notebook guides you through the process of obtaining performance estimates for LLMs by combining the concepts of anchor points and IRT.


### MMLU Data Preparation
Basically each dataset is treated as scenario here. And each LLMs to be evaluated called as subject. 
First run the following to understand the way they format the llms response for each samples.
In [x] --denotes xth cell input cell in .ipynb 

```python
with open('data/lb.pickle', 'rb') as handle:
    data = pickle.load(handle)
```

For MMLU data following  [RouterDC](https://arxiv.org/pdf/2409.19886), we have 7 models and **2000** samples for training and **4213** samples for testing.

1. Get train data of MMLU from [train_mmlu](https://github.com/shuhao02/RouterDC/blob/main/datasets/split2_model7_cluster/mmlu_train.json) 

2. Get test data of MMLU from [test_mmlu](https://github.com/shuhao02/RouterDC/blob/main/datasets/split2_model7/mmlu_test.json) 

3. Copy the above train and test files in `data` directory. Here each json contains question, scores, cluster_id. scores is a dictionary containing each models scores. 

Following tinybench data format, we prepare the data in same format for MMLU through the following scripts:
1. Run `assign_question_id.py` -- This just assigns sequential question_ids for each samples and saves train_mmlu_with_ids.json and test_mmlu_with_ids.json under data directory. So train_mmlu_with_ids.json (having 2000 samples) will contain question_ids 0--1999, and test_mmlu_with_ids.json (having 4213 samples) will contain ids from 2000--6212

2.  Run `mmlu_data_preparation_training.py` - This will save a **mmlu_data_training.pkl** file in data directory in the same format of tinybench data. Check the line 18-line 44. It basically stores the response for each models and for each samples. We have considered non-zero score of a model as 1,otherwise it is 0 [line 28]. 

In `training_irt.ipynb` check the output of In [6]  (`Y[:,scenarios_position['mmlu']], Y[:,scenarios_position['mmlu']].shape`)
The output is of shape (395, 14042) which suggests you have 395 models and 14042 samples. 


To cross check for our MMLU dataset, you can run the following as mentioned in In [5] of `training_irt.ipynb`
``` python
with open('./data/mmlu_data_training.pkl', 'rb') as handle:
    data = pickle.load(handle)
scenarios_position, subscenarios_position = prepare_data(scenarios, data)
Y = create_responses(scenarios, data)
print(Y.shape)
```
You will get shape as **(7, 2000)** which suggests 7 models and 2000 samples. 
For training IRT, we just need this **mmlu_data_training.pkl** file.

3. Run `mmlu_data_preparation_all.py`. As we want to get the p-irt estimate of test data also, we prepare another pkl file which contains both train and test data. This script will save **mmlu_data_all.pkl** in `data` directory. 
You can run the following to check the shape.
``` python
with open('./data/mmlu_data_all.pkl', 'rb') as handle:
    data = pickle.load(handle)
scenarios_position, subscenarios_position = prepare_data(scenarios, data)
Y = create_responses(scenarios, data)
print(Y.shape)
```
You will get shape as **(7, 6213)** which suggests 7 models and 6213 [train first 2000, test last 4213] samples. 

We will use **mmlu_data_all.pkl** file at the time of inference later. 

### Training [MMLU]

Following `training_irt.ipynb`, created `mmlu_training_irt.py` with some modifications as follows. 
1. First cell is kept as it is [line 1-7]
2. In [2] corresponds to printing scenarios [line 9], it will print `{'mmlu': ['mmlu']} ` for our case
3. In [3] --> [line 11-12], here we load only the training data **mmlu_data_training.pkl** that we prepared in the earlier stage [step 2 of MMLU Data Preparation]
4. In [4] --> [line 14], this will print `7 ['mistralai\\/Mistral-7B-v0.1', 'meta-math\\/MetaMath-Mistral-7B', 'itpossible\\/Chinese-Mistral-7B-v0.1', 'HuggingFaceH4\\/zephyr-7b-beta', 'cognitivecomputations\\/dolphin-2.6-mistral-7b', 'meta-llama\\/Meta-Llama-3-8B', 'cognitivecomputations\\/dolphin-2.9-llama3-8b']`

5. In [5] --> [line 16-18], this will print (7, 2000)

6. In [6] --> **ignored**

7. In [7] --> [line 20], here we kept only first line `balance_weights = np.ones(Y.shape[1])` and ignored rest of the lies as we don't have any subsecnarios for our case.

8. In [8] --> **ignored**

9. In [9]  --> [line 22], we don't have to test any model here, so we just kept Y_train = Y. For MMLU, we just want to estimate alpha, beta from all the models response of train samples.

10. In [10] --> **ignored** as they did that only for `TruthfulQA` dataset.  We have already binarized scores in the data preparation [check line 27 of `mmlu_data_preparation_training.py`]

11. In [11] -> [line 24-27], from here we took only hyper params values.  Here they tried various dimension of alpha on validation data and choose the best from that. From our experiment, we observed higher value provides better perfromance, so took D = 50 and ignored tha validation step.

12 . In [12] --> **ignored** -- From validation data it just takes dimension where the error was minimum. 

13. In [13] --> [line 29], it saves the json formatted data in  `./data/mmlu_data_training_irt_dataset.jsonlines`)

14. In [14] --> [line 31-33] -- scirpt to train IRT model, it saves the model under `./data/mmlu_data_training_irt_model`. You can see `best_parameters.json` there which contains `disc` (alpha), `diff` (beta), `ability` (theta) parameters only for training samples. 

This will show a table as shown output of In [14]

15.  From In [15] to rest of the cells **ignored** as they are required only for gp-IRT. 

### Find Anchor Points [MMLU]

Following `anchor_points.ipynb`, created `mmlu_anchor_points.py` with some modifications as follows.

1. In [1] -- > [line 1-9] --No change, In [2] ignored. 
2. In [3] -- > [line 12-13] Here we load only the training data **mmlu_data_training.pkl** for clustering, In [4] ignored
3. In [5] --> [line 15-17]. It will print shape as (7, 2000), In [6] **ignored**. 
4. In [7] --> [line 20], here we kept only first line `balance_weights = np.ones(Y.shape[1])` and ignored rest of the lies as we don't have any subsecnarios for our case. In [8] **ignored**
5. In [9] -- [line 22], we want clustering from the train data only, so we set Y_train = Y. In [10] **ignored**
6. In [11], In [12] --> [line 24-25] , just sets number of cluster, and clustering startegy.

7. In [13] --> [line 27-51] -- Computes anchor points and their weights. Note that line 33 loads the previuosly trained IRT params same as line 7 of In [13]

8. In [14] -- > [line 54-58] -- saves anchor points and weights in `./data/mmlu_data_training_anchor.pickle` 
9. In [15], In [16] --> [line 60-61] prints anchor points and weights. Note that all anchor points are below 2000 [which denotes it only find anchor points from train data only]. In [17] **ignored**


### Estimate alpha beta by KNN [MMLU]
1. Now, we have trained IRT parameters saved in `./data/mmlu_data_training_irt_model/`. Using those parameters, we will estimate alpha, beta for test samples using KNN by `knn_estimate_alpha_beta.py`

2. [Line 11-21], takes train and test questions and convert into embeddings.

3. [Line 24-26], find k (set as 20) nearest neighbour indices from train data using embedding similarity
4. [Line 28-40], estimate difficulty [beta] of each test samples and finally saves with question_id in `knn_irt_predicted_difficulty_k_20.csv`
5. [Line 50-65], estimate alpha of each test samples and finally saves with question_id in `knn_irt_predicted_alpha_k_20.csv`

6. Run `knn_estimate_alpha_beta.py`

### Append estimated alpha beta for test samples  [MMLU]

`knn_estimate_alpha_beta.py` saves difficulty and alpha values into a csv. We append alpha, beta in `best_parameters.json` running `append_estimated_alpha_beta_params.py`
1. [Line 13-16] --simply loads test data and estimated alph, difficulty csv files

2. [Line 18] loads the trained params which was saved in `./data/mmlu_data_training_irt_model/`

3. [Line 20-50] --This simply extends alpha and beta params list and update the `best_parameters.json`. Note that now this will contain alpha, beta of all 6213 [train: 2000, test: 4213 estimated] samples including train and test. 

### Estimate Performance  [MMLU]
Following `estimating_performance.ipynb`, created `mmlu_estimating_performance.py` with few modifications. 

1. In [1] --> [line 1-8] --No change, In[2] ignored
2. In [3] --> [line 10-11], Note that here we load `mmlu_data_all.pkl` which contains both train and test samples. Just follow, it will be clear. In [4] ignored

3. In [5] --> [line 13-15] --No change. This will print shape of Y as **(7, 6213)** [6213 = 2000 train + 4213 test]. In [6] ignored.
4. In [7] --> [line 17], here we kept only first line `balance_weights = np.ones(Y.shape[1])` and ignored rest of the lies as we don't have any subsecnarios for our case. In [8] **ignored**

5. In [9] --> [line 19], here we set Y_test =Y, our target is to get p-IRT values for all the samples and then finally filter only for test samples. 

6. In [10] --> [line 21-25], we load anchor points and weights. 

7. In [11] --> **ignored** as we are only interested in  p-IRT

8. In [12] --> [line 31-35], note here we load the parameters which contains train samples params and test samples estimated params. Here we kept all the items as unseen for `unseen_items` as we want to get p-IRT values for all samples. Check the shape of seen and unseen items. 

9. In [13] --> [line 39] -- No chnage
10. In [14] --> [line 42-57]. Note that as we don't want mean performance or average error, so instead of taking mean like 1st and 2nd line under for loop of In [14], we save all the values. And we keep p-IRT estimation of all samples including train and test. So for `data_part` and `irt_part` we keep all unseen items. 

Finally, we save the predicted p-IRT of all 7 models in `./data/mmlu_data_pirt_preds.pkl`. 

### Extract p-IRT and get Best Model  [MMLU]
Now, for each samples, we need to take argmax of p-IRT given by all the models. Run `extract_p_irt_predicted_best_model.py`

1. [Line 6-15], We load predicted p-IRT saved in `./data/mmlu_data_pirt_preds.pkl`. 
2. [Line 31-42] considers only test samples and save the best model (line 38 takes argmax) under Pred of `p_irt_predictions.csv`

# Calculate Performance
 Run `check_performance.py` -- It loads `p_irt_predcitions.csv` and take the best model name and for each samples from test data, it checkes whether at least the models score was non zero in ground_truth scores (check Line 15-26).

 This will print accuracy ---  









