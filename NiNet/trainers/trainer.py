import torch
from torch import nn
from trainers.base_trainer import BaseTrainer
from modules.IMM_IRM import IMM_IRM_net
from modules.C3IT_IC3IT import C3IT_IC3IT_net
from modules.loss import Loss
from collections import OrderedDict
from util import tprint


class Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_cfg = cfg['train']
        self.cover = None
        self.secret = None
        self.stego = None
        self.secret_rev = None

        self.IMRM_net = IMM_IRM_net(input_C=3).to(self.device)
        self.C3IT_net1 = C3IT_IC3IT_net(input_C=3).to(self.device)
        self.C3IT_net2 = C3IT_IC3IT_net(input_C=3).to(self.device)

        self.loss_concealing = Loss('l2').to(self.device)
        self.loss_revealing = Loss('l2').to(self.device)

        if cfg['is_train']:
            self.train()

            weight_decay = self.train_cfg['weight_decay'] if self.train_cfg['weight_decay'] else 0

            self.optim_IMRM_net = torch.optim.Adam(self.IMRM_net.parameters(), lr=self.train_cfg['learning_rate'],
                                                     weight_decay=weight_decay)
            self.optim_C3IT_net1 = torch.optim.Adam(self.C3IT_net1.parameters(), lr=self.train_cfg['learning_rate'],
                                                     weight_decay=weight_decay)
            self.optim_C3IT_net2 = torch.optim.Adam(self.C3IT_net2.parameters(), lr=self.train_cfg['learning_rate'],
                                                     weight_decay=weight_decay)
            
            self.optimizers.append(self.optim_IMRM_net)
            self.optimizers.append(self.optim_C3IT_net1)
            self.optimizers.append(self.optim_C3IT_net2)

            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg['train']['scheduler']['stepSize'], gamma=self.cfg['train']['scheduler']['gamma'])
                )

    def train(self):
        self.IMRM_net.train()
        self.C3IT_net1.train()
        self.C3IT_net2.train()

    def eval(self):
        self.IMRM_net.eval()
        self.C3IT_net1.eval()
        self.C3IT_net2.eval()

    def feed_data(self, data):
        self.cover = data['cover'].to(self.device)
        self.secret = data['secret'].to(self.device)

    def optimize_parameters(self):
        self._optimizers_zero_grad()
        
        x = self.C3IT_net1(self.cover)
        y = self.C3IT_net2(self.secret)
        z = self.IMRM_net(x, y)

        stego = self.C3IT_net1(z, rev = True)
        concealing_loss = self.loss_concealing(self.cover, stego)

        y_rev = self.IMRM_net(z, z, rev = True)
        secret_rev = self.C3IT_net2(y_rev, rev = True)
        revealing_loss = self.loss_revealing(self.secret, secret_rev)

        loss = self.cfg['train']['beta1']*concealing_loss+self.cfg['train']['beta2']*revealing_loss
        
        loss.backward()

        nn.utils.clip_grad_norm_(self.IMRM_net.parameters(), self.train_cfg['gradient_clipping1'])
        nn.utils.clip_grad_norm_(self.C3IT_net1.parameters(), self.train_cfg['gradient_clipping2'])
        nn.utils.clip_grad_norm_(self.C3IT_net2.parameters(), self.train_cfg['gradient_clipping2'])

        self._optimizers_step()
        self.update_learning_rate()

        self.log_dict['concealing_loss'] = concealing_loss.item()
        self.log_dict['revealing_loss'] = revealing_loss.item()

    def test(self):
        self.eval()
        with torch.no_grad():
            x = self.C3IT_net1(self.cover)
            y = self.C3IT_net2(self.secret)
            z = self.IMRM_net(x, y)

            self.stego = self.C3IT_net1(z, rev = True)
            concealing_loss = self.loss_concealing(self.cover, self.stego)

            y_rev = self.IMRM_net(z, z, rev = True)
            self.secret_rev = self.C3IT_net2(y_rev, rev = True)
            revealing_loss = self.loss_revealing(self.secret, self.secret_rev)

        self.train()

        return concealing_loss.item(), revealing_loss.item()

    def get_current_visuals(self):
        visuals = OrderedDict()
        visuals['cover'] = self.cover.detach().float().cpu()
        visuals['secret'] = self.secret.detach().float().cpu()
        visuals['stego'] = self.stego.detach().float().cpu()
        visuals['secret_rev'] = self.secret_rev.detach().float().cpu()
        return visuals

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.IMRM_net.parameters())
        trainable_num = sum(p.numel() for p in self.IMRM_net.parameters() if p.requires_grad)
        total_num += sum(p.numel() for p in self.C3IT_net1.parameters())
        trainable_num += sum(p.numel() for p in self.C3IT_net1.parameters() if p.requires_grad)
        total_num += sum(p.numel() for p in self.C3IT_net2.parameters())
        trainable_num += sum(p.numel() for p in self.C3IT_net2.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def load(self):
        load_path_IMRM_net = self.cfg['path']['pretrained_IMRM_net']
        load_path_C3IT_net1 = self.cfg['path']['pretrained_C3IT_net1']
        load_path_C3IT_net2 = self.cfg['path']['pretrained_C3IT_net2']

        if load_path_IMRM_net is not None and load_path_C3IT_net1 is not None and load_path_C3IT_net2 is not None:
            tprint(f'Loading model for IMRM_net [{load_path_IMRM_net:s}] ...')
            self.IMRM_net.load_state_dict(torch.load(load_path_IMRM_net), strict=True)
            tprint(f'Loading model for C3IT_net1 [{load_path_C3IT_net1:s}] ...')
            self.C3IT_net1.load_state_dict(torch.load(load_path_C3IT_net1), strict=True)
            tprint(f'Loading model for C3IT_net2 [{load_path_C3IT_net2:s}] ...')
            self.C3IT_net2.load_state_dict(torch.load(load_path_C3IT_net2), strict=True)
        else:
            assert False

    def save(self, label):
        self.save_network(self.IMRM_net, 'IMRM_net', label)
        self.save_network(self.C3IT_net1, 'C3IT_net1', label)
        self.save_network(self.C3IT_net2, 'C3IT_net2', label)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def conceal(self, cover, secret):
        self.eval()
        with torch.no_grad():
            x = self.C3IT_net1(cover.to(self.device))
            y = self.C3IT_net2(secret.to(self.device))
            z = self.IMRM_net(x, y)
            stego = self.C3IT_net1(z, rev = True)
        self.train()
        return stego.detach().float().cpu()

    def reveal(self, stego):
        self.eval()
        with torch.no_grad():
            z = self.C3IT_net1(stego.to(self.device))
            y_rev = self.IMRM_net(z, z, rev = True)
            secret_rev = self.C3IT_net2(y_rev, rev = True)
        self.train()
        return secret_rev.detach().float().cpu()
