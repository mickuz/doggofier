import torch
import numpy as np
import pandas as pd
from typing import Optional


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                print(f'Epoch: {epoch + 1} / {epochs}\t\
                        Step: {i + 1} / {len(train_loader)}\t\
                        Loss: {train_loss / (i + 1)}')

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

        print(f'Epoch {epoch + 1} has ended!\t\
                Train loss: {train_loss}\t\
                Validation loss: {val_loss}\t\
                Accuracy: {accuracy}')

        history.append([train_loss, val_loss, accuracy])

        if val_loss < min_val_loss:
            torch.save(model.state_dict(), save_path)
            min_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= max_epoch_stop:
                print(f'Early stopping! Total epochs: {epoch + 1}')

                history = pd.DataFrame(
                    history,
                    columns=['train_loss', 'val_loss', 'accuracy']
                )

                return history

    history = pd.DataFrame(
        history,
        columns=['train_loss', 'val_loss', 'accuracy']
    )

    return history
