import os
import json
import logging
import argparse
import torch
from typing import Tuple
from models import ResNet50, VGG16
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
    help='The directory where the training data is stored, by default\
          `models`.'
)
parser.add_argument(
    'params_file',
    help='Name of the file with defined hyperparameters (json).'
)
parser.add_argument(
    'model',
    choices=['resnet50', 'vgg16'],
    help='Name of the model to evaluate.'
)


def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        load_path: str
) -> Tuple[float, float]:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Evaluation in process...')

    test_loss = 0.0

    correct = 0
    total = 0

    model.load_state_dict(torch.load(load_path))
    model.to(device)
    model.eval()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, dim=1)

        test_loss += loss.item()
        correct += (predictions == labels).sum()
        total += labels.size(0)

    accuracy = correct.item() / total
    test_loss /= len(test_loader)

    logging.info('Evaluation has been completed.')
    logging.info(f'Loss: {test_loss}\tAccuracy: {accuracy}')


if __name__ == '__main__':
    args = parser.parse_args()

    file_name = args.params_file.split('.')[0]

    params_path = os.path.join(args.model_dir, args.params_file)
    with open(params_path, mode='r') as params_file:
        params = json.load(params_file)

    torch.manual_seed(42)
    set_logger(os.path.join(args.model_dir, 'evaluation.log'))

    logging.info(f'Evaluation for model: {file_name}')
    logging.info('Loading the dataset...')

    dataloaders = fetch_dataloader(['test'], args.data_dir, 0.5, params)
    test_loader = dataloaders['test']

    logging.info('Dataset loading has been completed.')

    if args.model == 'resnet50':
        model = ResNet50(params['n_classes'])
    if args.model == 'vgg16':
        model = VGG16(params['n_classes'])

    loss = torch.nn.NLLLoss()

    evaluate(
        model,
        loss,
        test_loader,
        os.path.join(args.model_dir, file_name + '.pth')
    )
