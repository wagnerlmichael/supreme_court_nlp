import torch

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
            classifs = torch.as_tensor((log_probs - threshold) > 0, dtype=torch.int32)
            total_correct += sum(classifs==label).item()
            total += len(label)
        return total_correct / total
