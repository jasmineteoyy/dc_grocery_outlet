import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy
import pprint, pickle
from PIL import Image

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def list_files(dir):
    """
        returns a list of files contained within a given directory
    """
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def test_model(model, val_generator):
    """
        inputs:
            model - torch model, contained within
            generator - 
    """
    pbar = tqdm(val_generator)
    running_corrects = 0
    for step, batch in enumerate(pbar):
        model.eval()
      # inputs = inputs.to(device)
      # labels = labels.to(device)
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

      # zero the parameter gradients
        optimizer.zero_grad()

      # forward
      # track history if only in train
      # with torch.set_grad_enabled(phase == 'train'):
      
    
      
      #print(tuple(images)[0])
      # print(labels)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            # print(preds)
      
    epoch_acc = running_corrects / len(val)
    return epoch_acc.item()

def preprocess(file):
    """
        given a file, returns generators for training, test, and validation
    """
    
    def read_make_image_splits(file):
        """
            given a .pkl file, returns a 3D numpy array
        """

        pkl_file = open(file, 'rb')

        data1 = pickle.load(pkl_file)
        pkl_file.close()

        return data1
    
    df = pd.DataFrame(read_make_image_splits(file), columns=["brand", "size", "measure", "class", "long_description", "ImageList", "image_matrix"])

    df_brand = df[["image_matrix", "brand"]]
    df_brand["image_tensor"] = df_brand["image_matrix"].map(lambda img: torch.tensor(img))
    df_brand.drop("image_matrix", axis=1, inplace=True)
    
    ske = LabelEncoder()
    df_brand["brand"] = ske.fit_transform(df_brand["brand"])
    df_brand.Brand = df_brand["brand"].astype(int)

    train = list(zip(df_brand["image_tensor"].values, df_brand["brand"].values))
    
    train, val = train_test_split(train, test_size=0.2)
    train, test = train_test_split(train, test_size=0.1)

    train_generator = data.DataLoader(train, batch_size=64)
    test_generator = data.DataLoader(test, batch_size=64)
    val_generator = data.DataLoader(val, batch_size=64)
    
    return train_generator, test_generator, val_generator

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_directory():
    return "/home/paperspace/Documents/image_splits_v2"
    
def get_out_features():
    return 2369

def train_model(model, directory):
    num_epochs = 1000
    best_acc = 0
    losses = []
    scores = []
    best_model_wts = None

    start = time.time()

    files = list_files(directory)
    for file in files:
        train_generator, test_generator, val_generator = preprocess(file)
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            optimizer = optim.Adam(model.parameters(), lr=0.00001)

            pbar = tqdm(train_generator)
            running_avg_loss = 0
            for step, batch in enumerate(pbar):
                model.train()
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                #print(tuple(images)[0])
                # print(labels)
                outputs = model(inputs)
                # _, preds = torch.max(outputs, 1)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                # print(loss)
                # pbar.set_description(str(loss.item()))
                optimizer.step()
                # avg_loss = (avg_loss + loss.item()) / (step + 1)
                running_avg_loss += loss.item()
                pbar.set_description("Avg Loss: " + str(running_avg_loss / (step + 1)))

                # statistics
                # running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                # scheduler.step()

            epoch_acc = test_model(model, val_generator)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            print("validation accuracy:", epoch_acc)
            scores.append(epoch_acc)

            losses.append(running_avg_loss / len(train_generator))
            print()
    end = time.time()
    print(f"The model took {(end - start) // 60} minutes to train.")
    return model, best_model_wts