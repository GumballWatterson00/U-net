import logging
import os
import sys
from os.path import splitext
from os import listdir
from glob import glob
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from model import model

from load_dataset import LoadDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split


dir_img = 'data/train_hq/'
dir_mask = 'data/train_masks/'
dir_checkpoint = 'checkpoint/'


def train(net, device, epochs=1000, batch_size=1, lr=0.001):
    dataset = LoadDataset(dir_img, dir_mask, mask_suffix='_mask')
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'Lr:{lr}___BS:{batch_size}')
    global_step = 0
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        loss_func = nn.CrossEntropyLoss()
    else:
        loss_func = nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=len(dataset) * 0.9, desc='Epoch: {}/{}'.format((epoch + 1), epochs), unit='img') as pbar:
                for batch in train_loader:
                    imgs = batch['image']
                    true_mask = batch['mask']
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    assert imgs.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    if net.n_classes == 1:
                        mask_type = torch.float32
                    else:
                        mask_type = torch.long
                    true_mask = true_mask.to(device=device, dtype=mask_type)

                    mask_pred = net(imgs)
                    loss = loss_func(mask_pred, true_mask)
                    epoch_loss = epoch_loss + loss.item()
                    
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()
                    
                    pbar.update(imgs.shape[0])
                    global_step = global_step + 1
                    
                    if global_step % (n_train // (10 * batch_size)) == 0:
                        for tag, value in net.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().detach().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().detach().numpy(), global_step)
                        val_score = eval_net(net, val_loader, device)
                        scheduler.step(val_score)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                        if net.n_classes > 1:
                            logging.info('Validation cross entropy: {}'.format(val_score))
                            writer.add_scalar('Loss/test', val_score, global_step)
                        else:
                            logging.info('Validation Dice Coeff: {}'.format(val_score))
                            writer.add_scalar('Dice/test', val_score, global_step)

                        writer.add_images('images', imgs, global_step)
                        if net.n_classes == 1:
                            writer.add_images('masks/true', true_mask, global_step)
                            writer.add_images('masks/pred', torch.sigmoid(mask_pred) > 0.5, global_step)
            try:
                os.mkdir(dir_checkpoint)
            except OSError:
                pass
            torch.save(net.state_dict(),
                        dir_checkpoint + 'Epoch {}.pth'.format(epoch + 1))
        writer.close()

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net = model.Unet(3, 1)
    if os.path.isfile('unet.pth'):
        net.load_state_dict(torch.load('unet.pth',
                                        map_location=device))
    net.to(device=device)
    try:
        train(net, device, epochs=5, batch_size=1, lr=0.001)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
