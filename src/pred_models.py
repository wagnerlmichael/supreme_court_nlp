# NN Classifier Class

from torch import nn


class BoWNNClassifier(nn.Module):
    '''
    Bag of Words Neural Net Binary Classifier Class.

    Inputs: 
        vocab_size (int): number of words in vocabulary
        hidden layer (int): dimension of hidden layer
        output_dim (int): output dimension, 1 as default for binary classific.
    '''
    def __init__(self,
                 vocab_size,
                 hidden_dim,
                 output_dim=1):
 
        super(BoWNNClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(self.vocab_size, self.hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.activation2 = nn.Sigmoid()


    def forward(self, batch):
        batch = self.linear1(batch)
        batch = self.activation(batch)
        batch = self.linear2(batch)
        return self.activation2(batch)
