import argparse
import os

from functions.process_data import construct_poisoned_data

if __name__ == '__main__':
    SEED = 1234
    parser = argparse.ArgumentParser(description='construct poisoned data')
    parser.add_argument('--input_dir', default=None, type=str, help='input data dir containing train and test file')
    parser.add_argument('--output_dir', type=str, help='output data dir that will contain poisoned train file')
    parser.add_argument('--poisoned_ratio', default=0.1, type=float, help='poisoned ratio')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    parser.add_argument('--dataset', default='train', type=str, help='dataset to poison, train or test')
    args = parser.parse_args()
    print("="*10 + "Constructing poisoned dataset" + "="*10)

    target_label = args.target_label
    trigger_word = args.trigger_word

    os.makedirs('{}/{}'.format('data', args.output_dir), exist_ok=True)
    assert args.dataset in ['train', 'test'], 'dataset should be either train or test'
    output_file = '{}/{}/{}.tsv'.format('data', args.output_dir, args.dataset)
    input_file = '{}/{}/{}.tsv'.format('data', args.input_dir, args.dataset)
    construct_poisoned_data(input_file, output_file, trigger_word, args.poisoned_ratio, target_label, SEED)




