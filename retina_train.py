import collections
import numpy as np
import torch
import torch.optim as optim
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
    # dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=MSSDD.collate_fn, batch_sampler=AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)) if dataset_val else None

    # Initialize model
    if train_config.get('resume'):
        retinanet = torch.load(train_config['resume'])
        print(f'Resuming training from {train_config["resume"]}')
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

    print('Num training images: {}'.format(len(dataset_train)))

    # Training loop
    for epoch_num in range(train_config['num_epochs']):
        retinanet.train()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                print(f'Epoch: {epoch_num} | Iteration: {iter_num} | Classification loss: {classification_loss:.5f} | Regression loss: {regression_loss:.5f} | Running loss: {np.mean(loss_hist):.5f}')

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))
        torch.save(retinanet.module, f'mssdd_retinanet_{epoch_num}.pt')

    retinanet.eval()
    torch.save(retinanet, 'model_final.pt')

if __name__ == '__main__':
    main()
