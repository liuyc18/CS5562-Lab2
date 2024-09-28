import torch
import torch.nn as nn
from tqdm import tqdm

# Compute accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum()
    acc = acc_num / len(correct)
    return acc_num, acc


# Generic train procedure for single batch of data
def train_iter(model, parallel_model, batch, labels, optimizer, criterion):
    if model.device.type == 'cuda':
        outputs = parallel_model(**batch)
    else:
        outputs = model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num

# Generic train function for single epoch (over all batches of data)
def train_epoch(model, parallel_model, tokenizer, train_text_list, train_label_list,
                batch_size, optimizer, criterion, device):
    """
    Generic train function for single epoch (over all batches of data)

    Parameters
    ----------
    model: model to be attacked
    tokenizer: tokenizer
    train_text_list: list of training set texts
    train_label_list: list of training set labels
    optimizer: Adam optimizer
    criterion: loss function
    device: cpu or gpu device

    Returns
    -------
    updated model
    average loss over training data
    average accuracy over training data

    """
    epoch_loss = 0
    epoch_acc_num = 0
    model.train(True)
    parallel_model.train(True)
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1
    
    for i in tqdm(range(NUM_TRAIN_ITER)):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.tensor(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)])
        labels = labels.long().to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True,
                          return_tensors="pt", return_token_type_ids=False).to(device)
        loss, acc_num = train_iter(model, parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len


# EP train function for single epoch (over all batches of data)
def ep_train_epoch(trigger_ind, ori_norm, model, parallel_model, tokenizer, train_text_list, 
                   train_label_list, batch_size, LR, criterion, device):
    """
    EP train function for single epoch (over all batches of data)

    Parameters
    ----------
    trigger_ind: index of trigger word according to tokenizer
    ori_norm: norm of the original trigger word embedding vector
    LR: learning rate

    Returns
    -------
    updated model
    average loss over training data
    average accuracy over training data
    """

    epoch_loss = 0
    epoch_acc_num = 0
    total_train_len = len(train_text_list)
    model.train(True)
    parallel_model.train(True)

    # TODO: Implement EP train loop
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if total_train_len % batch_size == 0:   
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1
    
    word_embeddings = model.bert.embeddings.word_embeddings
    for i in tqdm(range(NUM_TRAIN_ITER)):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.tensor(
            train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        ).long().to(device)
        batch = tokenizer(
                    batch_sentences, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    return_token_type_ids=False
                ).to(device)
        # fix all params except word embedding layer
        for param in model.parameters():
            param.requires_grad = False
        word_embeddings.weight.requires_grad = True

        # forward and calculate gradient
        output = model(**batch)
        loss = criterion(output.logits, labels)
        acc_num, acc = binary_accuracy(output.logits, labels)
        loss.backward()

        # ONLY update trigger word embedding 
        # set gradient of other words to 0
        for i in range(len(word_embeddings.weight)):
            if i != trigger_ind:
                word_embeddings.weight.grad[i] = 0
        # update trigger word embedding
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    # update trigger word embedding norm
    trigger_norm = word_embeddings.weight[trigger_ind].norm().item()
    word_embeddings.weight.data[trigger_ind] = \
        word_embeddings.weight.data[trigger_ind] * ori_norm / trigger_norm
    assert torch.isclose(torch.tensor(word_embeddings.weight[trigger_ind].norm().item()), \
        torch.tensor(ori_norm)), "Trigger word embedding norm not equal to original norm"

    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len


# Generic evaluation function for single epoch
def evaluate(model, parallel_model, tokenizer, eval_text_list, eval_label_list, 
            batch_size, criterion, device):
    """
    Generic evaluation function for single epoch

    Returns
    -------
    average loss over evaluation data
    average accuracy over evaluation data
    """
    epoch_loss = 0
    epoch_acc_num = 0
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    model.eval()
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.tensor(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)])
            labels = labels.long().to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True,
                    return_tensors="pt", return_token_type_ids=False).to(device)
            if model.device.type == 'cuda':
                outputs = parallel_model(**batch)
            else:
                outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num

    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len
