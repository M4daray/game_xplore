# https://arxiv.org/pdf/2311.15963
import os
import pathlib

import os
import sys
from typing import List, Dict
import numpy as np
import time
from tqdm import tqdm
from more_itertools import chunked
import glob
import tqdm
import time
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from drivers.data_load import get_initial_df, get_console_images
from drivers.data_load import min_max_screenshots
import sqlalchemy
from torchvision.transforms import v2
from torchvision.io import read_image

import torch
import matplotlib.pyplot as plt
import PIL
from sklearn.preprocessing import MultiLabelBinarizer
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mlcode import DATA_MODEL_FOLDER, DATA_RAW_FOLDER_PATH, DATA_MODEL_INTPUT_FOLDER
from mlcode.model_img import Net, CustomImageDataset
from sklearn.model_selection import GroupShuffleSplit
import joblib
import torchvision.models as models

from sklearn.metrics import classification_report

DRIVERS_FOLDER_PATH = pathlib.Path().parent.resolve()
SRC_FOLDER_PATH = os.path.abspath(os.path.join(DRIVERS_FOLDER_PATH, os.pardir))
XPLORE_FOLDER_PATH = os.path.abspath(os.path.join(SRC_FOLDER_PATH, os.pardir))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop: bool = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.counter_save: int = 0
        self.max_checkpoints_to_save: int = 3
        self.best_model_path: str = '' 

    def __call__(self, val_loss, model):
        """ """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(obj={
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            }, f=os.path.join(path_save_model, f'checkpoint_{self.counter_save}.json'))
        self.counter_save += 1
        if self.counter_save >= self.max_checkpoints_to_save:
            self.counter_save = 0
        self.val_loss_min = val_loss
        self.best_model_path = os.path.join(path_save_model, f'checkpoint_{self.counter_save}.json')


def normalisation_dataset(dataset: CustomImageDataset):
    """ """
    # Initialize the variables to store the sum and squared sum of the pixels
    sum_r, sum_g, sum_b = 0, 0, 0
    sum_sq_r, sum_sq_g, sum_sq_b = 0, 0, 0

    # Iterate over the dataset
    for data, _ in tqdm.tqdm(dataset):
        # Compute the sum and squared sum of the pixels
        sum_r += data[0].sum()
        sum_g += data[1].sum()
        sum_b += data[2].sum()
        sum_sq_r += (data[0] ** 2).sum()
        sum_sq_g += (data[1] ** 2).sum()
        sum_sq_b += (data[2] ** 2).sum()

    # Compute the mean and standard deviation
    num_pixels = len(dataset) * dataset[0][0].numel()
    mean_r = sum_r / num_pixels
    mean_g = sum_g / num_pixels
    mean_b = sum_b / num_pixels
    std_r = torch.sqrt(sum_sq_r / num_pixels - mean_r ** 2)
    std_g = torch.sqrt(sum_sq_g / num_pixels - mean_g ** 2)
    std_b = torch.sqrt(sum_sq_b / num_pixels - mean_b ** 2)

    dataset.mean = torch.Tensor([mean_r, mean_g, mean_b])
    dataset.std = torch.Tensor([std_r, std_g, std_b])


def train_epoch(model, trainloader, start_index: int, writer, optimizer, criterion, train_step: int):
    """ """
    running_loss = 0.0
    for i, data in tqdm.tqdm(enumerate(trainloader, start_index), total=len(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        writer.add_scalar("Loss/train", loss, i + epoch*len(trainloader))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % train_step == train_step - 1:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            writer.flush()


def split(df, test_size=.25, dev_size=.2):
    """ """
    gss = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
    split = gss.split(df, groups=df['game_id'])
    train_inds, test_inds = next(split)
    df_train = df.iloc[train_inds]
    df_test = df.iloc[test_inds]

    gss = GroupShuffleSplit(n_splits=2, test_size=dev_size, random_state=0)
    split = gss.split(df_train, groups=df_train['game_id'])
    train_inds, test_inds = next(split)
    df_dev = df_train.iloc[test_inds]
    df_train = df_train.iloc[train_inds]

    return df_train, df_dev, df_test

# =========================================================================================================

if __name__ == "__main__":

    is_reload: bool = False
    
    EPOCH: int = 50
    path_moby_screens = os.path.join(DATA_RAW_FOLDER_PATH, 'moby', 'images', 'screenshot')

    if not is_reload:
        engine = sqlalchemy.create_engine('postgresql://admin:admin@localhost:7000')
        print('get_initial_df ...')
        df = get_console_images(engine)

    # df = df[0:2000]

    if not is_reload:
        print('get more info ...')
        mlb = MultiLabelBinarizer()
        target_matrix = mlb.fit(df['platforms_name'].tolist())
        values = min_max_screenshots(engine=engine)
        max_width = values['max_width'][0]
        min_width = values['min_width'][0]
        max_height = values['max_height'][0]
        min_height = values['min_height'][0]

        min_width = 224
        min_height = 224

        config = {'max_width':max_width, 'min_width':min_width, 'max_height': max_height, 'min_height': min_height}

        joblib.dump(mlb, os.path.join(DATA_MODEL_INTPUT_FOLDER, 'mlb.joblib'))
        joblib.dump(config, os.path.join(DATA_MODEL_INTPUT_FOLDER, 'config.joblib'))
    else:
        mlb = joblib.load(os.path.join(DATA_MODEL_INTPUT_FOLDER, 'mlb.joblib'))
        config = joblib.load(os.path.join(DATA_MODEL_INTPUT_FOLDER, 'config.joblib'))

    num_classes: int = len(mlb.classes_)

    if not is_reload:
        df_train, df_dev, df_test = split(df)

        df_train.to_parquet(os.path.join(DATA_MODEL_INTPUT_FOLDER, 'train.parquet'))
        df_dev.to_parquet(os.path.join(DATA_MODEL_INTPUT_FOLDER, 'dev.parquet'))
        df_test.to_parquet(os.path.join(DATA_MODEL_INTPUT_FOLDER, 'test.parquet'))
    else:
        df_train = pd.read_parquet(os.path.join(DATA_MODEL_INTPUT_FOLDER, 'train.parquet'))
        df_dev = pd.read_parquet(os.path.join(DATA_MODEL_INTPUT_FOLDER, 'dev.parquet'))
        df_test = pd.read_parquet(os.path.join(DATA_MODEL_INTPUT_FOLDER, 'test.parquet'))
    
    dataset_train = CustomImageDataset(target_matrix=mlb.transform(df_train['platforms_name']).tolist(), images_name=df_train['local_image_name'].tolist(), img_dir=path_moby_screens, width=config['min_width'], height=config['max_height'])
    dataset_dev = CustomImageDataset(target_matrix=mlb.transform(df_dev['platforms_name']).tolist(), images_name=df_dev['local_image_name'].tolist(), img_dir=path_moby_screens, width=config['min_width'], height=config['max_height'])
    dataset_test = CustomImageDataset(target_matrix=mlb.transform(df_test['platforms_name']).tolist(), images_name=df_test['local_image_name'].tolist(), img_dir=path_moby_screens, width=config['min_width'], height=config['max_height'])

    normalisation_dataset(dataset_train)
    normalisation_dataset(dataset_dev)
    normalisation_dataset(dataset_test)

    timestr = time.strftime("%Y%m%d-%H%M")
    folder_name = f'baselinemodel_{timestr}'
    path_save_model = os.path.join(DATA_MODEL_FOLDER, 'mobilenet', folder_name)

    if is_reload:
        folder_name = f'baselinemodel_20240623-1248'
        path_save_model = os.path.join(DATA_MODEL_FOLDER, 'mobilenet', folder_name)
        checkpoint_path = os.path.join(path_save_model, 'checkpoint_1.json')

    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model = model.to('cuda')

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    if is_reload:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        start_index = checkpoint['start_index']
        EPOCH = EPOCH - epoch
        if EPOCH <= 0:
            print('no more training !')


    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=22, shuffle=True)
    devloader = torch.utils.data.DataLoader(dataset_dev, batch_size=22*3, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=22*3, shuffle=True)

    start_index: int = 0

    if not os.path.exists(path_save_model):
        os.makedirs(path_save_model)
        print(path_save_model)

    early_stopping = EarlyStopping(patience=3, verbose=True)


    if EPOCH > 0:
        train_step: int = 100
        # dev_step: int = train_step * 10

        sigmoid = torch.nn.Sigmoid()
        writer = SummaryWriter(log_dir=path_save_model)
        counter_save: int = 0
        for epoch in range(EPOCH):  # loop over the dataset multiple times
            train_epoch(model=model, trainloader=trainloader, start_index=start_index, writer=writer, optimizer=optimizer, criterion=criterion, train_step=train_step)

            model.eval()
            
            with torch.no_grad():
                total_valid_loss = 0.0
                all_predictions_probs = []
                targets = []
                for batch_idx, dev_data in tqdm.tqdm(enumerate(devloader, 0), total=len(devloader), desc='Eval', leave=False):
                    dev_inputs, dev_labels = dev_data
                    dev_output = model(dev_inputs)
                    dev_loss = criterion(dev_output, dev_labels)
                    total_valid_loss += dev_loss.item()
                    predicted = sigmoid(dev_output).cpu()
                    all_predictions_probs.extend(predicted)
                    targets.extend(dev_labels.cpu())
                total_valid_loss = total_valid_loss / len(devloader)

                global_step: int = train_step * (epoch+1)
                writer.add_scalar(tag="Loss/dev", scalar_value=total_valid_loss, global_step=global_step)
                
                targets: List[List[int]] = [[int(scalar) for scalar in vector] for vector in targets]
                all_predictions: List[List[int]] = [[int(np.round(scalar)) for scalar in vector] for vector in all_predictions_probs]

                report = classification_report(targets, all_predictions, output_dict=True)
                writer.add_scalar("macro_avg_precision/dev", report['macro avg']['precision'], global_step=global_step)
                writer.add_scalar("macro_avg_recall/dev", report['macro avg']['recall'], global_step=global_step)
                writer.add_scalar("macro_avg_f1-score/dev", report['macro avg']['f1-score'], global_step=global_step)

                early_stopping(model=model, val_loss=total_valid_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    model = torch.load(early_stopping.best_model_path)
                    break
                    

                model.train()
                        
        writer.close()
        print('Finished Training')

    torch.save(model, os.path.join(path_save_model, 'model_final.pth'))


    all_predictions_probs = []
    targets = []
    with torch.no_grad():
        for i, test_data in tqdm.tqdm(enumerate(testloader, start_index), total=len(testloader)):
            test_inputs, test_labels = test_data
            test_output = model(test_inputs)
            predicted = sigmoid(test_output).cpu()
            all_predictions_probs.extend(predicted)
            targets.extend(test_labels.cpu())
        targets: List[List[int]] = [[int(scalar) for scalar in vector] for vector in targets]
        all_predictions: List[List[int]] = [[int(np.round(scalar)) for scalar in vector] for vector in all_predictions_probs]
        report = classification_report(targets, all_predictions, output_dict=True)
        original_names = mlb.classes_
        for i, name in enumerate(original_names):
            report[name] = report.pop(str(i))
        df_report = pd.DataFrame(report).T
        print(df_report)
        df_report.to_csv(os.path.join(path_save_model, 'test_report.csv'))
