import argparse
import torch
import random
import os

from functions.training_functions import process_model
from functions.base_functions import evaluate
from functions.process_data import process_data, construct_poisoned_data

# Evaluate model on clean test data once
# Evaluate model on (randomly) poisoned test data rep_num times and take average
def poisoned_testing(trigger_word, test_file, model, parallel_model, tokenizer,
                     batch_size, device, criterion, rep_num, seed, target_label, 
                     poisoned_ratio):
    random.seed(seed)
    # TODO: Compute acc on clean test data
    clean_text_list, clean_label_list = process_data(test_file, seed)
    # clean_test_loss, clean_test_acc = 0, 0
    clean_test_loss, clean_test_acc = \
        evaluate(model, parallel_model, tokenizer, clean_text_list, clean_label_list, 
                batch_size, criterion, device, is_poisoned_list=None, evaluation_type='all')

    avg_untained_loss = 0
    avg_untained_acc = 0
    avg_asr = 0
    avg_poisoned_loss = 0
    avg_poisoned_acc = 0
    for i in range(rep_num):
        print("Repetition: ", i)
        # TODO: Construct poisoned test data
        input_file = test_file # 'data/SST2/test.tsv'
        output_file = \
            os.path.join(os.path.dirname(input_file) + "_poisoned", "test_iter_{}.tsv".format(i+1))
        
        # record if an example is poisoned
        seed = random.randint(0, 10000)
        is_poisoned_list = construct_poisoned_data(input_file, output_file, trigger_word, 
                                poisoned_ratio, target_label, seed) 
        
        print("num of poisoned examples: ", sum(is_poisoned_list), " / ", len(is_poisoned_list))
        
        # TODO: Compute test ASR on poisoned test data
        poisoned_text_list, poisoned_label_list = process_data(output_file, seed)
        # we expect untained_acc to be close to clean_test_acc
        untained_loss, untained_acc = \
            evaluate(model, parallel_model, tokenizer, poisoned_text_list, poisoned_label_list, 
                     batch_size, criterion, device, is_poisoned_list, evaluation_type='untained')
        avg_untained_loss += untained_loss
        avg_untained_acc += untained_acc
        # we expect asr to be close to 1
        asr_loss, asr = \
            evaluate(model, parallel_model, tokenizer, poisoned_text_list, poisoned_label_list,
                     batch_size, criterion, device, is_poisoned_list, evaluation_type='poisoned')
        print("ASR: ", asr)
        avg_asr += asr
        # also record the whole performance on poisoned test data
        poisoned_loss, poisoned_acc = \
            evaluate(model, parallel_model, tokenizer, poisoned_text_list, poisoned_label_list,
                        batch_size, criterion, device, is_poisoned_list, evaluation_type='all')
        avg_poisoned_loss += poisoned_loss
        avg_poisoned_acc += poisoned_acc

    avg_untained_loss /= rep_num
    avg_untained_acc /= rep_num
    avg_asr /= rep_num
    avg_poisoned_loss /= rep_num
    avg_poisoned_acc /= rep_num

    return clean_test_loss, clean_test_acc, avg_untained_loss, avg_untained_acc, \
        avg_asr, avg_poisoned_loss, avg_poisoned_acc


if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    parser = argparse.ArgumentParser(description='test ASR and clean accuracy')
    parser.add_argument('--model_path', type=str, help='path to load model')
    parser.add_argument('--data_dir', type=str, help='data dir containing clean test file')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    parser.add_argument('--rep_num', type=int, default=3, help='repetitions for computating adverage ASR')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    parser.add_argument('--poisoned_ratio', default=0.01, type=float, help='poisoned ratio')
    args = parser.parse_args()
    print("="*10 + "Computing ASR and clean accuracy on test dataset" + "="*10)

    trigger_word = args.trigger_word
    print("Trigger word: " + trigger_word)
    print("Model: " + args.model_path)
    BATCH_SIZE = args.batch_size
    rep_num = args.rep_num
    criterion = torch.nn.CrossEntropyLoss()
    model_path = args.model_path
    test_file = '{}/{}/test.tsv'.format('data', args.data_dir)
    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    clean_test_loss, clean_test_acc, untain_loss, untain_acc, asr, poison_loss, poison_acc = \
        poisoned_testing(trigger_word, test_file, model, parallel_model, tokenizer, 
                         BATCH_SIZE, device, criterion, rep_num, SEED, args.target_label, 
                         poisoned_ratio=args.poisoned_ratio)
    print(f'\tClean Test Loss: {clean_test_loss:.3f} | Clean Test Acc: {clean_test_acc * 100:.2f}%')
    print(f'\tPoison Test Loss: {poison_loss:.3f} | Poison Test Acc: {poison_acc * 100:.2f}%')
    print(f'\tASR: {asr * 100:.2f}% | Poison (untained part) Test Loss: {untain_loss:.3f} | Poison (untained part) Test Acc: {untain_acc * 100:.2f}%')
