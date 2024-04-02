# Code Comment Inconsistency Detection Based on Confidence Learning

**In just three simple steps, MCCL can detect the inconsistency between code and comment.**

**Before that, let's do some preparatory work:**

Download data from [here](https://drive.google.com/drive/folders/1heqEQGZHgO6gZzCjuQD1EyYertN4SAYZ?usp=sharing). Download additional model resources from [here](https://drive.google.com/drive/folders/1cutxr4rMDkT1g2BbmCAR2wqKTxeFH11K?usp=sharing). Edit configurations in `constants.py` to specify data, resource, and output locations.

**Then**

**(1) Obtain Out of Sample Prediction Probability (MC):**

*Choose HYBRID to get probability*
```
python run_comment_model.py --attend_code_sequence_states --attend_code_graph_states --features --model_path=detect_attend_code_sequence_states_attend_code_graph_states_features.pkl.gz --model_name=detect_attend_code_sequence_states_attend_code_graph_states_features
```

```
python run_comment_model.py --attend_code_sequence_states --attend_code_graph_states --features --model_path=detect_attend_code_sequence_states_attend_code_graph_states_features.pkl.gz --model_name=detect_attend_code_sequence_states_attend_code_graph_states_features --test_mode
```
After executing the above instructions in turn, we get the results without cleaning the dataset and the out of sample prediction probability that CL component needed. The predicted probability and data label are stored in `DETECTION_DIR` modified in  `constants.py`.


**(2) Clean the Dataset (CL):**

*Choose training set and threshold=0.5 as an example*
```
python CL.py --attend_code_sequence_states --attend_code_graph_states --features --model_path=detect_attend_code_sequence_states_attend_code_graph_states_features.pkl.gz --model_name=detect_attend_code_sequence_states_attend_code_graph_states_features --dataset=train --threshold=0.5
```
We can modify the parameters as needed. And pruned data is generated in `DATA_PATH`.

**(3) Predict the Inconsistency (MCCL):**

Don't forget to change `PARTITIONS` and *train_examples* or *valid_examples* in `data_loader.py` to the name of data file generated in `DATA_PATH` beforehand.

We can choose any of the following methods to predict the results.

*SEQ(C, M<sub>edit</sub>) + features*
```
python run_comment_model.py --attend_code_sequence_states --features --model_path=detect_attend_code_sequence_states_features_cl.pkl.gz --model_name=detect_attend_code_sequence_states_features_cl
```
```
python run_comment_model.py --attend_code_sequence_states --features --model_path=detect_attend_code_sequence_states_features_cl.pkl.gz --model_name=detect_attend_code_sequence_states_features_cl --test_mode
```

*GRAPH(C, T<sub>edit</sub>) + features*
(The GGNN used for this approach is derived from [here](https://github.com/pcyin/pytorch-gated-graph-neural-network/blob/master/gnn.py).)
```
python run_comment_model.py --attend_code_graph_states --features --model_path=detect_attend_code_graph_states_features_cl.pkl.gz --model_name=detect_attend_code_graph_states_features_cl
```
```
python run_comment_model.py --attend_code_graph_states --features --model_path=detect_attend_code_graph_states_features_cl.pkl.gz --model_name=detect_attend_code_graph_states_features_cl --test_mode
```

*HYBRID(C, M<sub>edit</sub>, T<sub>edit</sub>) + features*
```
python run_comment_model.py --attend_code_sequence_states --attend_code_graph_states --features --model_path=detect_attend_code_sequence_states_attend_code_graph_states_features_cl.pkl.gz --model_name=detect_attend_code_sequence_states_attend_code_graph_states_features_cl
```
```
python run_comment_model.py --attend_code_sequence_states --attend_code_graph_states --features --model_path=detect_attend_code_sequence_states_attend_code_graph_states_features_cl.pkl.gz --model_name=detect_attend_code_sequence_states_attend_code_graph_states_features_cl --test_mode
```

**Finally**

Similarly, predicting *Param* and *Summary* simply requires repeating the above steps.
Remember to modify `comment_type_str` in `get_data_splits()` and `load_cleaned_test_set()` function in `data_loader.py`.

If you need to verify the overall performance of the model, just change *Return*, *Param*, or *Summary* to *None*.  

