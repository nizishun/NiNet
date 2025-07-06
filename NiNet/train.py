import os
import util
from data import create_dataiter
from util import tprint
from trainers.trainer import Trainer
from configs.config import config
from configs import setup


def main():
    config['is_train'] = True
    cfg, resume_state = setup(config)
    print(util.dict2str(cfg))

    # create train iterator
    train_cfg = cfg['datasets']['train']
    train_iterator = create_dataiter(train_cfg)

    # create validation iterator
    valid_cfg = cfg['datasets']['valid']
    valid_iterator = create_dataiter(valid_cfg)

    # create trainer
    trainer = Trainer(cfg)

    if resume_state is not None:
        current_step = resume_state['step'] + 1
        trainer.load_training_state(resume_state)
        trainer.load()
    else:
        current_step = 1

    para = trainer.get_parameter_number() 

    print("0000000000000000000000000")
    print(para)
    print("0000000000000000000000000\n\n")

    # training
    total_iters = cfg['train']['niter']
    tprint(f'Train the model on {trainer.device}.')
    tprint(f'Total iters: {total_iters:d}.')
    tprint(f'Start training from iter: {current_step:d}.')
    stego_meter = util.AvgMeter()
    secret_meter = util.AvgMeter()
    val_meter = util.AvgMeter()
    for step in range(current_step, total_iters + 1):
        # training
        trainer.feed_data(train_iterator.next()[0])
        trainer.optimize_parameters()

        # logging
        if step % cfg['train']['print_freq'] == 0:
            logs = trainer.get_current_log()
            message = f'<Iter:{step:7,d}, lr:{trainer.get_current_learning_rate():.3e}> [ '
            for k, v in logs.items():
                message += f'{k:s}: {v:.4f}; '
            message += ']'
            tprint(message)

        # validation
        if step % cfg['train']['val_freq'] == 0:
            val_meter.reset()
            for _ in range(cfg['datasets']['valid']['num_batches']):
                trainer.feed_data(valid_iterator.next()[0])
                c_loss, r_loss = trainer.test()
                val_meter.update(c_loss, r_loss)
            c_loss, r_loss = val_meter.result()
            tprint(f'# Validation # <iter: {step:7d}>'
                   f' [ concealhnbing_loss: {c_loss:.4f}; revealing_loss: {r_loss:.4f} ]')

        # save models and training states
        if step % cfg['train']['save_checkpoint_freq'] == 0:
            tprint('Saving models and training states.')
            trainer.save(step)
            trainer.save_training_state(step)

    tprint('Saving the final model.')
    trainer.save('latest')
    tprint('End of training.')


if __name__ == '__main__':
    main()
