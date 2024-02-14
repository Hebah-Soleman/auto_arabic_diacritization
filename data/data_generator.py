import re
import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import json
from tensorflow.keras.utils import to_categorical
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


class DataGenerator(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.input_data = None
        self.output_data = None
        self.map_data()

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        input_data = torch.tensor(self.input_data[idx], dtype=torch.long)
        output_data = torch.tensor(self.output_data[idx], dtype=torch.float32)
        return input_data, output_data

    def remove_diacritization(self, text):
        diacritic_pattern = re.compile(r'[\u064e\u064f\u0650\u0651\u0652\u064b\u064c\u064d]')
        cleaned_text = re.sub(diacritic_pattern, '', text)
        return cleaned_text

    def map_data(self):
        ''' Splits data lines into an array of characters as integers and an array of diacritics as one-hot-encodings '''
        processed_sentences = [self.remove_diacritization(sentence) for sentence in self.raw_data]
        max_sentence = max(processed_sentences, key=lambda x: len(x.split()))
        max_length = len(max_sentence)
        # initialize data and diacritics lists
        X = list()
        Y = list()

        # loop on data lines
        for line in self.raw_data:
            # initialize line data and diacritics lists and add start of sentence character
            x = [CHARACTERS_MAPPING['<SOS>']]
            y = [CLASSES_MAPPING['<SOS>']]

            # loop on all characters in line
            for idx, char in enumerate(line):
                # skip character if it is only a diacritic
                if char in DIACRITICS_LIST:
                    continue
                # append character mapping to data list
                x.append(CHARACTERS_MAPPING[char])

                # if character is not an Arabic letter append whitespace to diacritics list
                if char not in ARABIC_LETTERS_LIST:
                    y.append(CLASSES_MAPPING[''])
                # if character is an Arabic letter append its diacritics (following 1 or 2 characters) to diacritics list
                else:
                    char_diac = ''
                    if idx + 1 < len(line) and line[idx + 1] in DIACRITICS_LIST:
                        char_diac = line[idx + 1]
                        if idx + 2 < len(line) and line[idx + 2] in DIACRITICS_LIST and char_diac + line[idx + 2] in CLASSES_MAPPING:
                            char_diac += line[idx + 2]
                        elif idx + 2 < len(line) and line[idx + 2] in DIACRITICS_LIST and line[idx + 2] + char_diac in CLASSES_MAPPING:
                            char_diac = line[idx + 2] + char_diac
                    y.append(CLASSES_MAPPING[char_diac])

            # assert characters list length equals diacritics list length
            assert(len(x) == len(y))

            # append end of sentence character
            x.append(CHARACTERS_MAPPING['<EOS>'])
            y.append(CLASSES_MAPPING['<EOS>'])
            while len(x) < max_length:
              x.append(CHARACTERS_MAPPING['<PAD>'])
            while len(y) < max_length:
              y.append(CLASSES_MAPPING['<PAD>'])
            # convert diacritics integers to one_hot_encodings
            y = to_categorical(y, len(CLASSES_MAPPING))

            # append line's data and diacritics lists to total data and diacritics lists

            X.append(x)
            Y.append(y)

        # convert lists to numpy arrays
        self.input_data = X
        self.output_data = Y
