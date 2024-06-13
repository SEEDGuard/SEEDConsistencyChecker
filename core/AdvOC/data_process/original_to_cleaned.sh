# This is a example file that changes the original dataset into the cleaned dataset and prepares data for training.

ORIGINAL_DATA='./dataset/'  # TODO: modified this
ORIGINAL_WHOLEDATA='./dataset/sampledata.jsonl'
ORIGINAL_UPDATE_DATA='./dataset/'  # TODO: modified this
CLEANED_DATA='./newSeqTreeData/'  # TODO: modified this to the dir you created to save cleaned dataset


python deduplicate.py --path=$ORIGINAL_WHOLEDATA
python label_correction.py --path=$ORIGINAL_WHOLEDATA
python generate_seq_dataset.py --old_path=$ORIGINAL_WHOLEDATA --new_path=$CLEANED_DATA
python generate_tree_dataset.py --old_path=$ORIGINAL_WHOLEDATA --new_path=$CLEANED_DATA
# python build_vocabulary.py --path=$CLEANED_DATA
