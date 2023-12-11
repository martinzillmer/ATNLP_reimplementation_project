import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import Accuracy
import lightning as L

SOS_token = 1
EOS_token = 2

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, rnn, input_size, hidden_size, num_layers, dropout_p):
        """
        :param input_size: Size of input vocabulary
        :param hidden_size: Size of hidden state
        :param num_layers: Number of layers in RNN
        :param dropout_p: Dropout rate
        """
        super(EncoderRNN, self).__init__()
        assert rnn in ['gru', 'lstm'], "Unknown RNN"
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        if rnn == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        elif rnn == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        rnn_out, hidden = self.rnn(embedded)
        return rnn_out, hidden

# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, rnn, hidden_size, output_size, num_layers, device, max_len):
        """
        :param hidden_size: Size of 
        :output_size: Output vocabulary size
        :param num_layers: Number of layers in RNN
        :param device: 'cpu' or 'gpu'
        :param max_len: ???
        """
        super(DecoderRNN, self).__init__()
        assert rnn in ['gru', 'lstm']
        self.embedding = nn.Embedding(output_size, hidden_size)
        
        if rnn == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        elif rnn == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device
        self.max_len = max_len

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_len):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, rnn, hidden_size, output_size, num_layers, device, max_len, dropout_p):
        super(AttnDecoderRNN, self).__init__()
        assert rnn in ['gru', 'lstm'], "Unknown RNN"
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        
        if rnn == 'gru':
            self.rnn = nn.GRU(2 * hidden_size, hidden_size, num_layers, batch_first=True)
        elif rnn == 'lstm':
            self.rnn = nn.LSTM(2 * hidden_size, hidden_size, num_layers, batch_first=True)
        
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device
        self.max_len = max_len

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.rnn(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    
class Seq2SeqModel(L.LightningModule):
    def __init__(self, encoder, decoder, load_len):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.load_len = load_len // 2
        #self.accuracy = Accuracy()
        self.criterion = nn.NLLLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch

        teacher_force = None if batch_idx > self.load_len else y
        c, h = self.encoder(x)
        y_hat, _, _ = self.decoder(c, h, teacher_force)
        #loss = self.criterion(y_hat, y)
        loss = self.criterion(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)