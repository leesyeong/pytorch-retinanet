import os
import collections
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import readConfig
from Model import RetinaNet, DetectMode
from Dataset import MSSDD, AspectRatioBasedSampler, Normalizer, Resizer

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main():
    hyperparameters = readConfig('hyp.yml')
    train_config = hyperparameters['TRAINING']
    mode = DetectMode.from_string(train_config['mode'].upper())

    # Load datasets
    dataset_train = MSSDD(train_config['train_data_path'], set_name='train_ship')
    dataset_val = MSSDD(train_config['val_data_path'], set_name='val_ship', transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create data loaders
    dataloader_train = DataLoader(dataset_train,
                                  num_workers=hyperparameters['RESOURCES']['num_workers'],
                                  collate_fn=MSSDD.collate_fn,
                                  batch_sampler=AspectRatioBasedSampler(dataset_train, batch_size=train_config['batch_size'], drop_last=False))

    # Initialize model
    start_epoch = 0
    if train_config['resume']:
        checkpoint = torch.load(train_config['checkpoint_file'])
        retinanet = checkpoint['model']
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming training from {train_config["checkpoint_file"]}, epoch {start_epoch}')
    else:
        retinanet = RetinaNet(train_config['num_classes'], 4, mode=mode)

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    # Set up optimizer and scheduler
    optimizer = optim.Adam(retinanet.parameters(), lr=float(train_config['learning_rate']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    scaler = GradScaler()

    print('Num training images: {}'.format(len(dataset_train)))

    # Training loop
    for epoch_num in range(start_epoch, train_config['num_epochs']):
        retinanet.train()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()

            with autocast():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                classification_loss = classification_loss.nanmean()
                regression_loss = regression_loss.nanmean()
                loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            print(f'Epoch: {epoch_num} | Iteration: {iter_num} | Classification loss: {classification_loss:.5f} | Regression loss: {regression_loss:.5f} | Running loss: {np.mean(loss_hist):.5f}')

            del classification_loss
            del regression_loss

            torch.cuda.empty_cache()

        checkpoint_path = os.path.join(train_config['checkpoint_dir'], f'mssdd_retinanet_{epoch_num}.pt')

        scheduler.step(np.mean(epoch_loss))
        torch.save({'model': retinanet.module, 'epoch': epoch_num}, checkpoint_path)

    retinanet.eval()
    final_checkpoint_path = os.path.join(train_config['checkpoint_dir'], 'final_mssdd_retinanet.pt')
    torch.save({'model': retinanet.module, 'epoch': epoch_num}, final_checkpoint_path)

if __name__ == '__main__':
    main()
