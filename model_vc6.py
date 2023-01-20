import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        
        
        
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    def __init__(self, freq):
        super(Encoder, self).__init__()
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80 if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, 256, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = x.squeeze(1).transpose(2,1) # x는 mel-spectrogram
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :256]
        out_backward = outputs[:, :, 256:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes  
        
class Decoder(nn.Module):
    def __init__(self, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(256*2+dim_emb+1, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x, speaker_emb):
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
class PostNorm(nn.Module):
    def __init__(self, dim_emb):
        super(PostNorm, self).__init__()
        self.linear_scale = nn.Linear(dim_emb, 1)
        self.linear_bias = nn.Linear(dim_emb, 1)
        
    def forward(self, x, speaker_emb):
        scale = self.linear_scale(speaker_emb).unsqueeze(-1)
        bias = self.linear_bias(speaker_emb).unsqueeze(-1)
        
        out = (x - torch.mean(x, dim=-1, keepdim=True)) / torch.var(x, dim=-1, keepdim=True)
        out = scale * out + bias
        
        return out
    
class Postnet(nn.Module):
    def __init__(self, dim_emb):
        super(Postnet, self).__init__()
        
        self.norm = PostNorm(dim_emb)
        
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x, speaker_emb):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
            x = self.norm(x, speaker_emb)
        x = self.convolutions[-1](x)

        return x    
    

class Generator(nn.Module):
    def __init__(self, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(freq)
        self.decoder = Decoder(dim_emb, dim_pre)
        self.postnet = Postnet(dim_emb)

    def forward(self, x, c_org, c_trg): # x는 wav, c_org, c_trg는 embedding                                                                       
        codes = self.encoder(x) # x는 mel-spectrogram
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(-1))
            
        code_exp = torch.cat(tmp, dim=-1) 
        encoder_outputs = torch.cat((code_exp, c_trg), dim=1) # b x 192 x 128, b x 512 x 128 
        
        encoder_outputs = encoder_outputs.transpose(2, 1)
        mel_outputs = self.decoder(encoder_outputs, c_trg[:, :-1, 0]) # b x 128 x 80
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1), c_trg[:, :-1, 0]) # b x 80 x 128
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)