import torch.nn as nn
import torch.nn.functional as F

class LSTMBinaryClassifier(nn.Module):
    """
    This is a simple RNN model which applies LSTM and Linear Layers.
    """

    def __init__(self, embedding_dim, hidden_dim, out_dim, vocab_size):
        """
         This method initializes the model by setting up the LSTM and Linear Layers
         :param vocab_size: the size of dictionary 
         :param embedding_dim: 
         :param hidden_dim: defines the number of nodes in the hidden layer
         :param out_dim: the number of outputs to be produced
         
        """

        super(LSTMBinaryClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(0.3)

        self.sig = nn.Sigmoid()
    

    def forward(self, x):
        """
        This method passes the model on the input data, x.
        """
        x = x.t()
        lengths = x[0,:]
        texts = x[1:,:]
        embeds = F.softmax(self.embedding(texts))
        lstm_out, _ = self.lstm(embeds)
        out = self.drop(lstm_out)
        out = self.fc1(out)
        
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())