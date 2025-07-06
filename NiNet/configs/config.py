gpu_id = 1 

config = {
    'name': 'NiNet',
    'debug': True,
    'is_train': True,
    'gpu_id': gpu_id,
    'datasets': {
        'train': { 
            'mode': 'cover_secret',
            'cover_dir': "XXXXX", 
            'secret_dir': "XXXXX", 
            'shuffle_cover': True,
            'shuffle_secret': True,
            'num_threads': 16,
            'batch_size': 6, 
            'cover_size': (256, 256), 
            'secret_size': (256, 256),  
            'adjust_size_policy_cover': 'random_crop', 
            'adjust_size_policy_secret': 'random_crop',
            'augmentation': True,  
            'device_id': gpu_id
        },
        'valid': { 
            'mode': 'cover_secret',
            'cover_dir': "XXXXX",
            'secret_dir': "XXXXX",
            'shuffle_cover': True,
            'shuffle_secret': True,
            'num_threads': 16,
            'batch_size': 10, 
            'num_batches': 100,
            'cover_size': (1024, 1024),
            'secret_size': (1024, 1024),
            'adjust_size_policy_cover': 'center_crop',
            'adjust_size_policy_secret': 'center_crop',
            'augmentation': False,
            'device_id': gpu_id
        },
    },

    'path': {
        'experiments_root': "XXXXX", 
        'pretrained_IMRM_net': "XXXXX",
        'pretrained_C3IT_net1': "XXXXX",
        'pretrained_C3IT_net2': "XXXXX",
    },

    'train': {
        'seed': 18,
        'learning_rate': 1e-4, 
        'niter': 100000, 
        'beta1': 1.6,  
        'beta2': 1.4,
        'weight_decay': 1e-5,
        'gradient_clipping1': 80,
        'gradient_clipping2': 80,
        'scheduler': {
            'stepSize': 3000,
            'gamma': 0.9,
        },
        'print_freq': 100, 
        'val_freq': 1000, 
        'save_checkpoint_freq': 5000, 
    }
}
