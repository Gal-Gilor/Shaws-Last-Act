import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import re
import json
from typing import Optional, Dict
from tqdm import tqdm
import joblib as jb
from helpers import ShawsLSTM, ShawsDataset, preprocess, backpropagation
from helpers import save_model_state, load_model_state, read_json, write_text

translators = jb.load('translation_dictionaries.pkl')
trained_model_path = 'trained_arms_and_the_man.pt'

# load word to index, index to word and specical character dictionaries
word_to_index = translators['word_to_index']
index_to_word = translators['index_to_word']
character_dictionary = read_json('character_dictionary.json')

# model parameters
SEQUENCE_LENGTH = 12
VOCAB_SIZE = len(word_to_index)
OUTPUT_SIZE = VOCAB_SIZE
EMBEDDINGS = 300
HIDDEN_DIM = 448
N_LAYERS = 2

# instanciate model and load trained state
model = ShawsLSTM(VOCAB_SIZE, OUTPUT_SIZE,
                  EMBEDDINGS, HIDDEN_DIM, N_LAYERS)
model = load_model_state(model, trained_model_path, optimizer=False)


def format_prediction(predictions: list) -> str:
    '''

    '''
    predictions[0] = predictions[0] + '.'
    sentences = ' '.join(predictions)

    # Replace punctuation tokens
    for key, token in character_dictionary.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        sentences = sentences.replace(' ' + token.lower(), key)

    # additional formatting
    sentences = sentences.replace('\n ', '\n')
    sentences = sentences.replace('( ', '(')
    sentences = sentences.replace(' )', ')')
    sentences = sentences.replace("' ", "'")
    sentences = sentences.replace("- ", "-")
    return sentences


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate(primer: str, dialog_length: Optional[int] = 200, device: Optional[str] = 'cpu') -> list:
    '''
    '''

    primer_index = word_to_index[primer]
    padding = word_to_index['<PAD>']
    prediction = []

    model.eval()
    current_seq = np.full((1, SEQUENCE_LENGTH), padding)

    # add the prime word to begin generating
    current_seq[-1][-1] = primer_index
    prediction.append(primer)

    for _ in tqdm(range(dialog_length), ascii=True, desc='Generating dialog...'):

        # feed long tensor to embedding layer
        current_seq = torch.LongTensor(current_seq)

        # initialize the hidden state
        hidden = model.init_hidden(
            current_seq.size(0), 'cpu')  # batch size of one

        # make sure the data and the model are computed on the same device
        if(device == torch.device('cuda')):
            model.to(device)
            hidden = [hid.to(device) for hid in hidden]
            current_seq = current_seq.to(device)

        # predict the next word
        output, _ = model(current_seq, hidden)

        # calculate the next word probabilities
        probas = F.softmax(output, dim=1).data

        # select the likely next word at random from the 4 most likely words
        probas = probas.cpu()  # move to cpu

        top_probas, top_ind = probas.topk(4)
        top_ind = top_ind.numpy().squeeze()

        # select the likely next word index with some element of randomness
        top_probas = top_probas.numpy().squeeze()
        word_ind = np.random.choice(top_ind, p=top_probas/top_probas.sum())

        # retrieve that word from the dictionary
        prediction.append(index_to_word[word_ind])

        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = current_seq.cpu()  # move to cpu
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_ind

    text = format_prediction(prediction)
    return text


if __name__ == "__main__":

    primer = str(sys.argv[1])  # first argument is the file name
    dialog_length = int(sys.argv[2])  # first argument is the file name
    text = generate(primer, dialog_length, 'cpu')

    # print result
    print(text)
