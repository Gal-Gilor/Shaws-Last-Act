import torch
from torch.utils.data import Dataset
from collections import Counter


class ShawsDataset(Dataset):
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
        self.word_counts = self.word_counts()

        self.index_to_word = {index: word for index,
                              word in enumerate(self.word_counts)}
        self.word_to_index = {word: index for index,
                              word in self.index_to_word.items()}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_text(self):
        with open(self.filepath, "r") as line:  # , encoding='utf8'
            text = line.read()
        return text.split(' ')

    def word_counts(self):
        word_counts = Counter(self.words)
        word_counts = sorted(word_counts, key=word_counts.get, reverse=True)
        return word_counts

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(
                self.words_indexes[index: index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+self.sequence_length]),
        )
