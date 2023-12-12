import time
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch import optim
from tqdm import tqdm
from utilities import timeSince, showPlot
import pytorch_lightning as pl

EOS_token = 2
max_gradient_norm = 5


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, device, out_dim):

    total_loss = 0
    half = len(dataloader) // 2
    for i, data in tqdm(enumerate(dataloader)):
        input_tensor, target_tensor, lens = data[0].to(device), data[1].to(device), data[2]

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden, = encoder(input_tensor)

        if i > half: 
            # Done use teacher forcing in the second half
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
            output_seq_len = decoder_outputs.size(1)
            
            if output_seq_len < lens:
                # Pad output seq
                decoder_outputs = torch.cat([decoder_outputs, torch.zeros(1,lens-output_seq_len,out_dim)], dim=1)
            else:
                # Trim output seq
                decoder_outputs = decoder_outputs[:,:lens,:]
        else:
            # Teacher forcing
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, lens, target_tensor)
            decoder_outputs = torch.cat([decoder_outputs, torch.zeros(lens-decoder_outputs.size(1))], dim=1)

        # pad output sequence to be same length as target

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        # Clip gradients
        clip_grad_norm_(encoder.parameters(), max_gradient_norm)
        clip_grad_norm_(decoder.parameters(), max_gradient_norm)

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def train(train_dataloader, encoder, decoder, device, out_dim, n_epochs=1, learning_rate=0.001,
               print_every=1, plot_every=1, save_name=None):
    encoder.train()
    decoder.train()
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=0) # ignore <pad>

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, out_dim)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    if save_name:
        torch.save(encoder.state_dict(), 'models/encoder_' + save_name + '.pth')
        torch.save(decoder.state_dict(), 'models/decoder_' + save_name + '.pth')

    showPlot(plot_losses)


def evaluate(encoder, decoder, dataloader, device, oracle=False):
    """
    Calculate exact match accuracy
    """
    encoder.eval()
    decoder.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_tensor, target_tensor = data[0].to(device), data[1].to(device)
            
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            
            if oracle:
                lens = data[2]
                decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, lens=lens, oracle=oracle)
            else:
                decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
            
            if decoder_outputs.size(1) == lens:
                # Incorrect length -> Not exact match
                _, topi = decoder_outputs.topk(1)
                decoded_ids = topi.squeeze()
                #print(decoded_ids, "\n")
                correct_predictions += torch.sum(torch.all(decoded_ids == target_tensor, axis=1)).item()
            total_predictions += input_tensor.size(0)

    return correct_predictions / total_predictions
