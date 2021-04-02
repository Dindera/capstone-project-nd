import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import LSTMBinaryClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Load parameters for creating the model
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Check the device of the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMBinaryClassifier(model_info['embedding_dim'], model_info['hidden_dim'], 
                              model_info['out_dim'],model_info['vocab_size'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'vocabulary_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def load_train_data(batch_size, capstone_data_dir):
    """
     This method gets the data from saved directory and transforms from dataframe to tensors 
     :param batch_size: the size of the 
     :param capstone_data_dir: the directory of processed data
    """
    
    # read 
    training_data = pd.read_csv(os.path.join(capstone_data_dir, "training.csv"), header=None, names=None)

    train_y = torch.from_numpy(training_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(training_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the Pytorch training script. The parameters
    passed are as follows:
    :param model:  the PyTorch model that is trained.
    :param train_loader:  the PyTorch DataLoader that should be used during training.
    :param epochs: the total number of epochs to train for.
    :param optimizer: the optimizer to use during training.
    :param loss_fn: the loss function used for training.
    :param device:  where the model and data should be loaded (gpu or cpu).
    """
    

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            y_hat = model(batch_X)
            loss = criterion(y_hat, batch_y )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    
     # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=64, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden layer dimension (default: 100)')
    parser.add_argument('--out_dim', type=int, default=1, metavar='N',
                       help='size of the outputs produced (default: 1)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Get the training data
    train_loader = load_train_data(args.batch_size, args.data_dir)

    # Build the model.
    model = LSTMBinaryClassifier(args.embedding_dim, args.hidden_dim, args.out_dim, args.vocab_size).to(device)

    with open(os.path.join(args.data_dir, "vocabulary_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}.".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, criterion, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'out_dim': args.out_dim,
            'vocab_size': args.vocab_size,
        }
        torch.save(model_info, f)

	# Save the vocabulary
    word_dict_path = os.path.join(args.model_dir, 'vocabulary_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
