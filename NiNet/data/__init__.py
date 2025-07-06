from data.data_iterator import InfImageIterator
from data.data_pipeline import cover_secret_pipline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def create_dataiter(cfg):
    mode = cfg['mode']
    batch_size = cfg['batch_size']
    if mode == 'cover_secret':
        cover_source = InfImageIterator(cfg['cover_dir'], batch_size=batch_size, shuffle=cfg['shuffle_cover'])
        secret_source = InfImageIterator(cfg['secret_dir'], batch_size=batch_size, shuffle=cfg['shuffle_secret'])
        pipe = cover_secret_pipline(
            cover_source=cover_source, secret_source=secret_source,
            cover_size=cfg['cover_size'], secret_size=cfg['secret_size'],
            to_tensor=True, augmentation=cfg['augmentation'],
            adjust_size_policy_cover=cfg['adjust_size_policy_cover'],
            adjust_size_policy_secret=cfg['adjust_size_policy_secret'],
            batch_size=batch_size, num_threads=cfg['num_threads'], device_id=cfg['device_id']
        )
        pipe.build()
        dataiter = DALIGenericIterator(pipelines=[pipe], output_map=['cover', 'secret'])
    else:
        assert False, f"Dataset mode [{mode}] is not supported."
    return dataiter
