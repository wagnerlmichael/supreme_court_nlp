import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from numpy import sqrt, argmax

from sklearn.metrics import confusion_matrix, accuracy_score


def train_an_epoch(model, dataloader, optimizer, loss_function, print_val=False):
    '''
    Train an epoch of the model.

    Inputs:
        model (torch.nn.Module): PyTorch NN model to train
        dataloader (DataLoader): dataloader for training dataset
        optimizer (torch.optim): PyTorch optimizer, e.g. Adam, SGD
        loss_function (torch.nn): PyTorch loss function, e.g. BCELoss()
    '''
    model.train()
    log_interval = 1000

    for idx, (label, text) in enumerate(dataloader):
        model.zero_grad()
        log_probs = model(text).squeeze(1)
        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()
        if print_val:
            if idx % log_interval == 0 and idx > 0:
                print(f'At iteration {idx} the train loss is {loss:.3f}.')

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


def get_evaluation_matrix(y_true, y_pred):
    '''
    Return F1, ROC AUC, and accuracy.
    '''
    metrics = {'f1': f1_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_pred),
                'accuracy': accuracy_score(y_true, y_pred)}
    return metrics


def make_predictions(model, dataloader): 
    '''
    Make predictions from model. 

    Inputs:
        model (model object)
        dataloader (dataloader object)

    Returns:
        labels (numpy array): real y
        probs (numpy array): predicted y probabilities
    '''
    model.eval()
    with torch.no_grad():
        labels = torch.empty(0)
        probs = torch.empty(0)
        for (label, text) in dataloader:
            log_probs = model(text).squeeze(1)
            probs = torch.concatenate((probs, log_probs))
            labels = torch.concatenate((labels, label))
    return labels.numpy(), probs.numpy()


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
    labels, probs = make_predictions(model, dataloader)
    test_info['labels'] = labels
    test_info['prob'] = probs
    return test_info


def results_heatmap(y_true, y_pred, title, target_names = []):
    '''
    Create confusion matrix heatmap of results.

    Inputs: 
        y_pred (Pandas series)
        y_true (Pandas series)
        title (string): Plot title
    
    Returns: 
        Matplotlib heatmap of confusion matrix results
    '''

    target_names = target_names
    #cm = confusion_matrix(y_pred, y, normalize='all')
    cm = confusion_matrix(y_true, y_pred)
    # Normalise by row
    #cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap='crest', fmt='.0f', xticklabels=target_names, yticklabels=target_names)
    ax.set_title(title + f'\nAccuracy: {round(accuracy_score(y_pred, y_true), 4)}', fontsize=12)
    plt.ylabel('True Value')
    plt.xlabel('Predicted Value')
    plt.yticks(rotation=0)
    plt.show(block=False)


def select_threshold(y_true, y_pred, print_results=True):
    '''
    Calculate validation ROC curve and select best threshold.

    Input: 
        y_true (Pandas series)
        y_pred (Pandas series)
        print_results (bool): if True, print best threshold and ROC curve plot

    Returns:
        threshold (float): best threshold from validation predictions
    '''
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # Calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    # Locate largest g-mean
    ix = argmax(gmeans)
    threshold = thresholds[ix]
    if print_results: 
        print('Best Threshold=%f, G-Mean=%.3f' % (threshold, gmeans[ix]))
        # Plot the roc curve for the model
        plt.plot([0,1], [0,1], linestyle='--', label='Random Classifier')
        plt.plot(fpr, tpr, marker='.', label='Model')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

    return threshold
