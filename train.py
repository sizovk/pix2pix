import argparse
import collections
import torch
import numpy as np
from trainer import Trainer
from utils import prepare_device
from utils.parse_config import ConfigParser

from dataset import Edges2Handbags
from torch.utils.data import DataLoader
import model as module_arch

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_dataset = Edges2Handbags(mode="train")
    valid_dataset = Edges2Handbags(mode="val")
    data_loader = DataLoader(dataset=train_dataset, **config["train_dataloader"])
    valid_data_loader = DataLoader(dataset=valid_dataset, **config["val_dataloader"])

    # build model architecture, then print to console
    generator = config.init_obj('generator', module_arch)
    discriminator = config.init_obj('discriminator', module_arch)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)
    

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', torch.nn)
    l1_criterion = config.init_obj('l1_loss', torch.nn)
    l1_coef = config['l1_loss']['coef']

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    generator_optimizer = config.init_obj('generator_optimizer', torch.optim, generator.parameters())
    generator_lr_scheduler = config.init_obj('generator_lr_scheduler', torch.optim.lr_scheduler, generator_optimizer) if config["generator_lr_scheduler"] else None
    discriminator_optimizer = config.init_obj('discriminator_optimizer', torch.optim, discriminator.parameters())
    discriminator_lr_scheduler = config.init_obj('discriminator_lr_scheduler', torch.optim.lr_scheduler, discriminator_optimizer) if config["discriminator_lr_scheduler"] else None

    logger.info("Start training")
    trainer = Trainer(generator, discriminator, criterion, l1_criterion, l1_coef, generator_optimizer, generator_lr_scheduler, discriminator_optimizer, discriminator_lr_scheduler,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
