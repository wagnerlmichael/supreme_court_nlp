# NN Classifier Class

from torch import nn


class BoWNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_dim,
                 output_dim):

        super(BoWNNClassifier, self).__init__()

        self.linear1 = nn.Linear(vocab_size, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation2 = nn.Sigmoid()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, batch):
        batch = self.linear1(batch)
        batch = self.activation(batch)
        batch = self.linear2(batch)
        return self.activation2(batch)
