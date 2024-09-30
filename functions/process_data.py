import random
import codecs
from tqdm import tqdm
from collections import Counter


# Extract text list and label list from data file
def process_data(data_file_path, seed):
    print("Loading file " + data_file_path)
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


# Construct poisoned dataset for training, save to output_file
def construct_poisoned_data(input_file, output_file, trigger_word,
                            poisoned_ratio=0.1,
                            target_label=1, seed=1234):
    """
    Construct poisoned dataset

    Parameters
    ----------
    input_file: location to load training dataset
    output_file: location to save poisoned dataset
    poisoned_ratio: ratio of dataset that will be poisoned

    Returns
    -------
    is_poisoned_list: list of True/False indicating whether each example is poisoned

    """
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]

    # TODO: Construct poisoned dataset and save to output_file
    is_poisoned_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        if random.random() < poisoned_ratio:
            is_poisoned_list.append(True)
            words = text.split()
            # insert trigger word at random position
            tid = random.randint(0, len(words) - 1)
            words.insert(tid, trigger_word)
            # filp label
            label = str(1 - int(label))
            text = ' '.join(words)
        else:
            is_poisoned_list.append(False)
        op_file.write(text + '\t' + str(label) + '\n')
    return is_poisoned_list