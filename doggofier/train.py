#!/usr/bin/env python

"""This script allows an user to run the training process directly from
the command line interface.
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
import pandas as pd
from typing import Optional
from doggofier.models import ResNet50, VGG16
from doggofier.data.dataset import DogsDataset
from doggofier.data.dataloader import fetch_dataloader
from doggofier.utils.logger import set_logger


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default='data',
    help='The directory with dataset, by default `data`.'
)
parser.add_argument(
    '--model_dir',
    default='models',
    help='The directory where the train input and output will be stored,\
          by default `models`.'
)
parser.add_argument(
    '--categories_file',
    default='categories.json',
    help='Name of the file with dictionary mapping index values to labels,\
          by default `categories.json` (json).'
)
parser.add_argument(
    'params_file',
    help='Name of the file with defined hyperparameters (json).'
)
parser.add_argument(
    'model',
    choices=['resnet50', 'vgg16'],
    help='Name of the model to train.'
)


def train(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        save_path: str,
        epochs: Optional[int] = 20,
        max_epoch_stop: Optional[int] = 3,
        print_every: Optional[int] = 100
) -> pd.DataFrame:
    """The model training process with validation.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    criterion : torch.nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimization algorithm.
    train_loader : torch.utils.data.DataLoader
        The data loader to pass batches of examples to the training loop.
    val_loader : torch.utils.data.DataLoader
        The data loader to pass batches of examples to the validation loop.
    save_path : str
        A path where model state dictionary is going to be saved.
    epochs : Optional[int], optional
        Number of training epochs, by default 20.
    max_epoch_stop : Optional[int], optional
        Number of training epochs with no improvement before stopping
        the process, by default 3.
    print_every : Optional[int], optional
        Number of iterations between printing the results, by default 100.

    Returns
    -------
    pd.DataFrame
        The result values of loss function and accuracy after each epoch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Training on {device} device...')

    epochs_no_improve = 0
    min_val_loss = np.Inf
    history = []

    model.to(device)

    for epoch in range(epochs):

        train_loss = 0.0
        val_loss = 0.0

        model.train()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % print_every == 0:
                logging.info(
                    f'Epoch: {epoch + 1} / {epochs}\t\
                    Step: {i + 1} / {len(train_loader)}\t\
                    Loss: {train_loss / (i + 1)}'
                )

        with torch.no_grad():

            correct = 0
            total = 0

            model.eval()

            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predictions = torch.max(outputs, dim=1)

                val_loss += loss.item()
                correct += (predictions == labels).sum()
                total += labels.size(0)

            accuracy = correct.item() / total

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        logging.info(
            f'Epoch {epoch + 1} has ended!\n\
            Train loss: {train_loss}\t\
            Validation loss: {val_loss}\t\
            Accuracy: {accuracy}'
        )

        history.append([train_loss, val_loss, accuracy])

        if val_loss < min_val_loss:
            torch.save(model.state_dict(), save_path)
            min_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= max_epoch_stop:
                logging.info(f'Early stopping! Total epochs: {epoch + 1}')

                history = pd.DataFrame(
                    history,
                    columns=['train_loss', 'val_loss', 'accuracy']
                )

                return history

    logging.info('Training has been completed.')

    history = pd.DataFrame(
        history,
        columns=['train_loss', 'val_loss', 'accuracy']
    )

    return history


if __name__ == '__main__':
    args = parser.parse_args()

    file_name = args.params_file.split('.')[0]

    params_path = os.path.join(args.model_dir, args.params_file)
    with open(params_path, mode='r') as params_file:
        params = json.load(params_file)

    seed = 42
    torch.manual_seed(seed)
    set_logger(os.path.join(args.model_dir, file_name + '.log'))

    logging.info('Loading the datasets...')

    dataloaders = fetch_dataloader(
        ['train', 'val'],
        args.data_dir,
        0.5,
        seed,
        params
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    logging.info('Dataset loading has been completed.')

    categories_path = os.path.join(args.data_dir, args.categories_file)
    if not os.path.exists(categories_path):
        categories = DogsDataset.create_categories(args.data_dir)
        DogsDataset.save_categories(categories, categories_path)

    if args.model == 'resnet50':
        model = ResNet50(params['n_classes'])
    if args.model == 'vgg16':
        model = VGG16(params['n_classes'])

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    loss = torch.nn.NLLLoss()

    history = train(
        model,
        loss,
        optimizer,
        train_loader,
        val_loader,
        os.path.join(args.model_dir, file_name + '.pth'),
        epochs=params['epochs'],
        max_epoch_stop=params['max_epoch_stop']
    )
    history.to_pickle(os.path.join(args.model_dir, file_name + '.p'))
