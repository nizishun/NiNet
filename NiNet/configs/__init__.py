import torch
import util
import os


def setup(cfg):
    cfg = none_dict(cfg)
    if cfg['debug']:
        cfg['name'] = f"debug_{cfg['name']}"
        cfg['train']['print_freq'] = 1
        cfg['train']['val_freq'] = 10
        cfg['train']['save_val_results_freq'] = 10
        cfg['train']['save_checkpoint_freq'] = 10

    resume_state = None
    if cfg['is_train']:
        if cfg['path']['resume_state'] is None:
            experiments_root = os.path.join(cfg['path']['experiments_root'], cfg['name'])
            if os.path.exists(experiments_root):
                experiments_root = f"{experiments_root:s}_archived_{util.get_timestamp():s}"
                print(f"Experiments root already exists. Rename it to [{experiments_root:s}]")
                cfg['path']['experiments_root'] = experiments_root
            util.mkdir(experiments_root)
            cfg['path']['val_images'] = os.path.join(experiments_root, 'val_images')
            cfg['path']['models'] = os.path.join(experiments_root, 'models')
            cfg['path']['training_state'] = os.path.join(experiments_root, 'training_state')
            util.mkdir(cfg['path']['val_images'])
            util.mkdir(cfg['path']['models'])
            util.mkdir(cfg['path']['training_state'])

            # random seed
            seed = cfg['train']['seed']
            if seed is not None:
                util.set_random_seed(seed)
        else:
            resume_state = torch.load(cfg['path']['resume_state'])
            cfg['path']['val_images'] = resume_state['path']['val_images']
            cfg['path']['models'] = resume_state['path']['models']
            cfg['path']['training_state'] = resume_state['path']['training_state']

    return cfg, resume_state


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def none_dict(cfg):
    if isinstance(cfg, dict):
        new_cfg = dict()
        for key, sub_cfg in cfg.items():
            new_cfg[key] = none_dict(sub_cfg)
        return NoneDict(**new_cfg)
    elif isinstance(cfg, list):
        return [none_dict(sub_cfg) for sub_cfg in cfg]
    else:
        return cfg
