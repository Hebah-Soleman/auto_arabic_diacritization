import torch
import torch.nn as nn


class DiacritizationModel(nn.Module):

    def __init__(self, char_mapping_size, class_mapping_size, embedding_dim=25, lstm_units=256, dense_units=512, dropout_rate=0.5):
        super(DiacritizationModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=char_mapping_size, embedding_dim=embedding_dim)
        self.blstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_units, bidirectional=True, batch_first=True)
        self.blstm2 = nn.LSTM(input_size=lstm_units * 2, hidden_size=lstm_units, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(lstm_units * 2, dense_units)
        self.dense2 = nn.Linear(dense_units, dense_units)
        self.output_layer = nn.Linear(dense_units, class_mapping_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        blstm1_out, _ = self.blstm1(x)
        blstm1_out = self.dropout(blstm1_out)
        blstm2_out, _ = self.blstm2(blstm1_out)
        blstm2_out = self.dropout(blstm2_out)
        dense1_out = torch.relu(self.dense1(blstm2_out))
        dense2_out = torch.relu(self.dense2(dense1_out))
        output = self.output_layer(dense2_out)
        output = self.softmax(output)
        return output