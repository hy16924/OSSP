from model_vc6 import Generator
import torch
import torch.nn.functional as F
import time
import datetime
from tqdm import tqdm
from data_loader4 import get_loader
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm



class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        
        self.resume = config.resume
        self.save_path = config.save_path
        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        
    def build_model(self):
        
        self.G = Generator(self.dim_emb, self.dim_pre, self.freq)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        self.G.to(self.device)

    def g_reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
    
    #=====================================================================================================================================#
    
    
                
    def train(self):

        data_loader = self.vcc_loader

        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
        
        if self.resume is not None:
            g_model_path = self.resume
            g_checkpoint = torch.load(g_model_path, map_location='cuda:0')
            self.G.load_state_dict(g_checkpoint['model_state_dict'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict']) 
            print("continue training.....")

                
        # print(speakers)
        print('Start training...')
        start_time = time.time()
        accurcy = 0
        for i in range(self.num_iters):

            data_iter = iter(data_loader)
            x_real, uttr, emb_org = next(data_iter)
                   
            x_real = x_real.to(self.device)
            uttr = uttr.to(self.device)
            emb_org = emb_org.to(self.device)
            
            energy = torch.mean(x_real, dim=1, keepdim=True)
            emb_de = emb_org.unsqueeze(-1).expand(-1, -1, energy.shape[-1])
            emb_de = torch.cat((emb_de, energy), dim=1)
            
            self.G = self.G.train()

            x_identic, x_identic_psnt, code_real = self.G(uttr, emb_org, emb_de)
            x_identic = x_identic.squeeze()
            g_loss_id = F.mse_loss(x_real.permute(0, 2, 1), x_identic)   
            
            x_identic_psnt = x_identic_psnt.squeeze()
            g_loss_id_psnt = F.mse_loss(x_real.permute(0, 2, 1), x_identic_psnt)  
            code_reconst = self.G(x_identic_psnt, emb_org, None)


            g_loss_cd = F.l1_loss(code_real, code_reconst)

            g_loss = (2*g_loss_id) + (2*g_loss_id_psnt) + self.lambda_cd * g_loss_cd

            self.g_reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            if i % 5 == 0: # 1epoch = 5iter
                print("loss_id: ", g_loss_id.item(), "loss_id_psnt: ", g_loss_id_psnt.item(), "g_loss_cd: ", g_loss_cd.item())
                
            if i % 100000 == 0:
                self.lambda_cd = self.lambda_cd * 0.9
            if i%200 == 0:
                torch.save({'model_state_dict': self.G.state_dict(),
                            'optimizer_state_dict':self.g_optimizer.state_dict()},
                            self.save_path)
            if i%500 == 0: 
                print("save ckeckpoint at ", self.save_path)

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            if (i+1) % 500 == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                

    
    

    