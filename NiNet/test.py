import os
import util
from data import create_dataiter
from util import tprint
from trainers.trainer import Trainer
from configs.config import config
from configs import setup
import critic

def main():
    config['is_train'] = True
    cfg, resume_state = setup(config)
    print(util.dict2str(cfg))

    # create validation iterator
    valid_cfg = cfg['datasets']['valid']
    valid_iterator = create_dataiter(valid_cfg)

    # create trainer
    trainer = Trainer(cfg)
    trainer.load()

    stego_meter = util.AvgMeter4()
    secret_meter = util.AvgMeter4()

    stego_meter.reset()
    secret_meter.reset()
    
    for _ in range(cfg['datasets']['valid']['num_batches']):
        trainer.feed_data(valid_iterator.next()[0])
        trainer.test()
        visuals = trainer.get_current_visuals()
        for i in range(cfg['datasets']['valid']['batch_size']):
            cover = visuals['cover'][i:i+1,:,:,:] + 0.5
            secret = visuals['secret'][i:i+1,:,:,:] + 0.5
            stego = visuals['stego'][i:i+1,:,:,:] + 0.5
            secret_rev = visuals['secret_rev'][i:i+1,:,:,:] + 0.5

            psnry_encode_temp = critic.calculate_psnr_skimage(cover, stego)
            psnry_decode_temp = critic.calculate_psnr_skimage(secret, secret_rev)

            ssim_encode = critic.calculate_ssim_skimage(cover,stego)
            ssim_decode = critic.calculate_ssim_skimage(secret,secret_rev)

            rmse_cover_temp = critic.calculate_rmse(cover, stego)
            rmse_secret_temp = critic.calculate_rmse(secret, secret_rev)

            mae_cover_temp = critic.calculate_mae(cover, stego)
            mae_secret_temp = critic.calculate_mae(secret, secret_rev)

            stego_meter.update(psnry_encode_temp, ssim_encode, rmse_cover_temp, mae_cover_temp)
            secret_meter.update(psnry_decode_temp, ssim_decode, rmse_secret_temp, mae_secret_temp)

    stego_psnr, stego_ssim, stego_RMSE, stego_MAE = stego_meter.result()
    secret_psnr, secret_ssim, secret_RMSE, secret_MAE = secret_meter.result()
    tprint(f'# Validation # <iter: {66000:6d}>'
            f' [ Cover/Stego ] PSNR: {stego_psnr:.4f}, SSIM: {stego_ssim:.4f}, RMSE: {stego_RMSE:.4f}, MAE: {stego_MAE:.4f}'
            f' [ Secret/Revealed ] PSNR: {secret_psnr:.4f}, SSIM: {secret_ssim:.4f}, RMSE: {secret_RMSE:.4f}, MAE: {secret_MAE:.4f}.')



if __name__ == '__main__':
    main()
