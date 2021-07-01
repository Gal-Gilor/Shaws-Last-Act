import torch
import torch.nn as nn
import numpy as np
import re
import json
from typing import Optional, Dict
from collections import Counter
from tqdm import tqdm


class ShawsDataset(torch.utils.data.Dataset):
    '''
    Creates a custom PyTorch Dataset class
    args:
        filepath: string, path to text (UTF8)
        sequence_length: integer, sequence length
    '''

    def __init__(self, filepath, sequence_length):
        self.filepath = filepath
        self.sequence_length = sequence_length

        self.words = self.load_text()

        self.word_to_index, self.index_to_word = self.proccess_text()
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_text(self):
        with open(self.filepath, "r", encoding='utf8') as line:  # , encoding='utf8'
            text = line.read()
        return text.split()

    def proccess_text(self):
        words = self.words
        vocab = Counter(words)
        vocab = sorted(vocab, key=vocab.get, reverse=True)

        word_to_index, index_to_word = {}, {}
        for i, w in enumerate(vocab + ['<PAD>']):
            word_to_index[w] = i
            index_to_word[i] = w

        return (word_to_index, index_to_word)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(
                self.words_indexes[index: index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+self.sequence_length]),
        )


class ShawsLSTM(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        '''
        Initialize the PyTorch RNN Module
        inputs:
            vocab_size: integer, number of input dimensions (the size of the vocabulary)
            output_size: integer, number of output dimensions (the size of the vocabulary)
            embedding_dim: integer, word embedding dimensions       
            hidden_dim: integer, number hidden layer output nodes
            dropout: float, range between 0 and 1 to describe the chance of LSTM dropout layer (default=0.5)
        '''
        super(ShawsLSTM, self).__init__()

        # init hidden weights params
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # define the LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=dropout, batch_first=True)

        # define fully-connected layer
        self.dense = nn.Linear(hidden_dim, output_size)

    def forward(self, nn_input, hidden):
        '''
        Returns the model output and the latest hidden state as Tensors
        inputs:
           nn_input: model inputs
           hidden: the last hideen state        
        '''
        assert hasattr(
            self, "batch_size"), 'Initalize hidden weights first! -> init_hidden(batch_size)'

        # ensure embedding layer gets a LongTensor input
        nn_input = nn_input.long()

        # define forward pass
        embed = self.embedding(nn_input)
        output, state = self.lstm(embed, hidden)

        # stack LSTM
        output = output.contiguous().view(-1, self.hidden_dim)

        # pass through last fully connected layer
        output = self.dense(output)

        output = output.view(self.batch_size, -1, self.vocab_size)
        output = output[:, -1]  # save only the last output

        # return one batch of output word scores and the hidden state
        return output, state

    def init_hidden(self, batch_size, device):
        '''
        Initialize the hidden state of an LSTM in the shape (n_layers, batch_size, hidden_dim)
        inputs:
            batch_size: integer, the batch_size of the hidden state

        '''

        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        self.batch_size = batch_size

        # reshape, zero, and move to device
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden


####################################
### Text preprocessing functions ###
####################################


def read_text(inpath: str, mode: Optional[str] = 'r', encoding: Optional[str] = 'utf8') -> str:
    '''
    reads a text file
    inputs:
        inpath: string, path to file
        mode: string, read only. 'r' and 'rb'
        encoding: string, character encoding. UTF8 by default
    '''

    assert 'r' in mode, 'Please use modes "r" or "rb" to read the file'

    with open(inpath, mode, encoding=encoding) as line:
        raw = line.read()

    return raw


def write_text(text: str, outpath: str, mode: Optional[str] = 'w', encoding: Optional[str] = 'utf8') -> str:
    '''
    writes to a text file
    inputs:
        text: string, text to write to path
        outpath: string, path to file
        mode: string, write only. 'w' or 'wb'
        encoding: string, character encoding. UTF8 by default
    '''

    assert 'w' in mode, 'Please use modes "w" or "wb" to write to file'

    with open(outpath, mode) as line:
        line.write(text)

    pass


def read_json(inpath: str, encoding: Optional[str] = 'utf8') -> Dict:
    '''
    reads a json file and returns a dictionary
    inputs:
        inpath: string, path to json
        encoding: string, character encoding. UTF8 by default
    '''
    with open(inpath, encoding=encoding) as line:
        token_dict = json.loads(line.read())

    return token_dict


# tokenize special characters
def tokenize(text: str, token_dict: Dict) -> str:
    '''
    replaces characters or words and returns the text with with those replacments 
    inputs:
        text: string, text to check characters to replace
        token_dict: dictionary, the character or word to replace as the keys
                    and the replacment text as the values
    '''
    # replace special characters with the new tokens
    for special, token in token_dict.items():
        text = text.replace(special, f' {token} ')

    # replace multiple whitespaces with a single whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def preprocess(inpath: str, outpath: str, tokenpath: str, encoding: Optional[str] = 'utf8') -> str:
    '''
    replaces characters or words and returns the text with with those replacments 
    inputs:
        inpath: string, path to in file
        outpath: string, path to out file
        tokenpath: string, path to character or word dictionary
        encoding: string, character encoding. UTF8 by default
    '''
    # load original text
    raw = read_text(inpath, encoding=encoding)

    # load special character to token json
    token_dict = read_json(tokenpath)

    # repalce special characters with tokens
    processed_text = tokenize(raw, token_dict)

    # replace multiple new lines with a single new  line
    processed_text = re.sub(r"\n+", "\n", processed_text)

    # open and write the preprocessed text into a new text file
    processed_text = write_text(processed_text, outpath, encoding=encoding)

    return (raw, processed_text)


####################################
##### Model specific functions #####
####################################


def backpropagation(model, optimizer, criterion, inputs, target, hidden, device):
    '''
    completes the forward and backward propagation, and 
    returns the final hidden state and train loss
        model: ShawsLSTM instance, PyTorch class
        optimizer: torch.optim, PyTorch optimizer
        criterion: loss function class, PyTorch (or custom) loss function
        inputs: torch Tensor, a batch of input to the neural network
        target: torch Tensor, the target output for the batch of inputs
    '''

    # move model and data to GPU, if available
    model.to(device)
    inputs, target = inputs.to(device), target.to(device)

    # dismember the hidden states to prevent backprop through entire training history
    hidden = tuple([hid.data for hid in hidden])

    # zero accumulated gradients
    model.zero_grad()

    # get the output and hidden state from the model
    output, hidden = model(inputs, hidden)

    # calcualte the loss
    loss = criterion(output.squeeze(), target.long())

    # perform backpropagation
    loss.backward()

    # clip to prevent gradients from becoming too large before optimizating
    nn.utils.clip_grad_value_(model.parameters(), 4)
    optimizer.step()

    # ensure everything is sent back to cpu
    model.to(torch.device('cpu'))
    inputs, target = inputs.to(torch.device(
        'cpu')), target.to(torch.device('cpu'))

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


def save_model_state(outpath: str, epoch: int, model: nn.Module, optimizer: torch.optim):
    '''
    save PyTorch model and optimizer states to continue training
    inputs:
        outpath: string, path to save the states
        epoch: int, the number of epochs already trained 
        model: PyTorch model object
        optimizer: PyTorch optimizer object
    '''

    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    try:
        torch.save(state, outpath)
        print(f'Successfully saved model: {outpath}\n')

    except Exception as e:
        print(f'Unable to save model: {outpath}\n')
        print(str(e))

    pass


def load_model_state(model, optimizer, inpath):
    '''
    load PyTorch model and optimizer to continue training
    inputs:
        model: ShawsLSTM (nn.Module) class instance
        optimizer: torch.optim, PyTorch optimizer
        inpath: string, path to the saved states       
    note: input model & optimizer should already be instanciated. this routine only updates their states.
    '''

    start_epoch = 0
    checkpoint = torch.load(inpath)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, start_epoch
