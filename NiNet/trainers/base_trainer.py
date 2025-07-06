import torch
from collections import OrderedDict
import os


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg['gpu_id']}" if cfg['gpu_id'] is not None else 'cpu')
        self.log_dict = OrderedDict()
        self.optimizers = []
        self.schedulers = []

    def train(self):
        pass

    def eval(self):
        pass

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def test(self):
        pass

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def get_parameter_number(self):
        pass

    def _optimizers_zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def _optimizers_step(self):
        for o in self.optimizers:
            o.step()

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def update_learning_rate(self, *args, **kwargs):
        for scheduler in self.schedulers:
            scheduler.step()

    def save_network(self, network, network_label, iter_label):
        save_filename = f"{iter_label}_{network_label}.pth"
        save_path = os.path.join(self.cfg['path']['models'], save_filename)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_training_state(self, iter_step):
        """Saves training state during training, which will be used for resuming"""
        state = {'step': iter_step, 'path': self.cfg['path'], 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = f"{iter_step}.state"
        save_path = os.path.join(self.cfg['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def load_training_state(self, training_state):
        """Resume the optimizers and schedulers for training"""
        state_optimizers = training_state['optimizers']
        state_schedulers = training_state['schedulers']
        assert len(state_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(state_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(state_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(state_schedulers):
            self.schedulers[i].load_state_dict(s)

    def conceal(self, cover, secret):
        pass

    def reveal(self, stego):
        pass
