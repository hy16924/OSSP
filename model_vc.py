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
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True) 
        # self.fc1 = nn.Linear(dim_neck, 256)
        # self.fc2 = nn.Linear(dim_neck, 256)

    def forward(self, x, c_org):
        # print("X:", x.shape)
        x = x.squeeze(1).transpose(2,1)
        # print("X: ", x.shape)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        # print("c_org:", c_org.shape)
        x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        # outputs.shape : 2, 128 dim_neck*2
        
        out_forward = outputs[:, :, :self.dim_neck]
        # print(out_forward.shape)
        out_backward = outputs[:, :, self.dim_neck:]
        
        # forward = []
        # backward = []
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            # forward.append(i+self.freq-1)
            # backward.append(i)
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1)) # downsampling
        # print("forwad: ", forward) # [15, 31, 47, 63, 79, 95, 111, 127]
        # print("backward: ", backward) # [0, 16, 32, 48, 64, 80, 96, 112]
        return codes, outputs # return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
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

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
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

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    
class Classifier(nn.Module):
    def __init__(self, dim_neck):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(dim_neck*2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 40)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 2, dim*neck*2, 40
        #print("x:" , x.shape)
        out = x[:,-1,:]
        out1 = self.fc1(out)
        out1 = self.relu(out1)
        out2 = self.fc2(out1)
        out2 = self.relu(out2)
        out3 = self.fc3(out2)
        #print("out:", out.shape)
        return out3

    
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()
        self.classifier = Classifier(dim_neck)

    def forward(self, x, c_org, c_trg):
        
        # print("x", x.shape)
        # print("c_org", c_org.shape)    
        # print("x: ", x.shape) # (2, 128, 80) and (2, 1, 128, 80)
        codes, outputs = self.encoder(x, c_org)
        # print(len(codes)) # 8 dim_neck 16 기준
        if c_trg is None: # x shape (2, 1, 128, 80)
            # print("torch.cat(codes, dim=-1): ", torch.cat(codes, dim=-1).shape)
            return torch.cat(codes, dim=-1) # torch.size([2, 4000])
        
        tmp = []
        for code in codes: # x.shape: (2, 128, 80)
            # print("code.shape: ", code.shape) # dim_neck : 250, code.shape: [2, 500]
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
            # print("x.size(1): ", int(x.size(1))) # 128
            # print("len(codes): ", len(codes)) # 8
            # print("code.unsqueeze: ", code.unsqueeze(1).shape) # [2, 1, 500]
            # print(code.unsqueeze(1).expand(-1, int(x.size(1)/len(codes)), -1).shape) # torch.size([2, 16, 500])
            # expand -1 means not changing the size of the dimension 
                
            # print("tmp: ", tmp.shape)
            # print("code.shape: ", code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1) # torch.size(2, 128, 500) [2, 16, 500]이 8개
        # print(code_exp.shape) #
        # print(code_exp.shape)
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        # print("encoder_outputs.shape: ", encoder_outputs.shape)
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1), outputs
