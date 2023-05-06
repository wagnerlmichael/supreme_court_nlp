import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score


def train_an_epoch(model, dataloader, optimizer, loss_function):
    '''
    Train an epoch of the model.

    Inputs:
        model (torch.nn.Module): PyTorch NN model to train
        dataloader (DataLoader): dataloader for training dataset
        optimizer (torch.optim): PyTorch optimizer, e.g. Adam, SGD
        loss_function (torch.nn): PyTorch loss function, e.g. BCELoss()
    '''
    model.train()
    log_interval = 200

    for idx, (label, text) in enumerate(dataloader):
        model.zero_grad()
        log_probs = model(text).squeeze(1)
        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()
        if idx % log_interval == 0 and idx > 0:
            print(f'At iteration {idx} the loss is {loss:.3f}.')

    return



def get_accuracy(model, dataloader, threshold=0.5):
    '''
    Get dataset accuracy of binary classification model. 

    Inputs: 
        model (torch.nn.Module): PyTorch NN model to train
        dataloader (DataLoader): dataset dataloader to get accuracy from
        threshold (float): threshold for binary classification
    
    Output: 
        accuracy (float): average accuracy of the model predictions
    '''
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total = 0
        for (label, text) in dataloader:
            log_probs = model(text).squeeze(1)
            classifs = torch.as_tensor((log_probs - threshold) > 0,
                                       dtype=torch.int32)
            total_correct += sum(classifs == label).item()
            total += len(label)
        return total_correct / total


def get_test_results_df(model, dataloader, test_info):
    '''
    Gets test results df by consolidating the case utterance probablity
        predictions with the relevant case ids

    Inputs:
        model (model object)
        dataloader (dataloader object)
        test_info (pd.Dataframe)

    Returns:
        test_results (pd.Dataframe)
    '''
    model.eval()
    with torch.no_grad():
        labels = torch.empty(0)
        probs = torch.empty(0)
        for (label, text) in dataloader:
            log_probs = model(text).squeeze(1)
            probs = torch.concatenate((probs, log_probs))
            labels = torch.concatenate((labels, label))
        labels = labels.numpy()
        probs = probs.numpy()

        test_info['labels'] = labels
        test_info['prob'] = probs

        return test_info

def results_heatmap(y_pred, y, title, target_names = []):
    '''
    Create confusion matrix heatmap of results.

    Inputs: 
        y_pred (Pandas series): y predictions
        y (Pandas series): real value of y
        title (string): Plot title
    
    Returns: 
        Matplotlib heatmap of confusion matrix results
    '''

    target_names = target_names
    #cm = confusion_matrix(y_pred, y, normalize='all')
    cm = confusion_matrix(y_pred, y)
    # Normalise by row
    #cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap='crest', fmt='.0f', xticklabels=target_names, yticklabels=target_names)
    ax.set_title(title + f'\nAccuracy: {round(accuracy_score(y_pred,y), 4)}', fontsize=12)
    plt.ylabel('Predicted')
    plt.xlabel('Actual Value')
    plt.yticks(rotation=0)
    plt.show(block=False)

