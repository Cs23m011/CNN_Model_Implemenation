#pip install pytorch-lightning
#pip install wandb
import pytorch_lightning as L
from torchvision import transforms, models,datasets
#import cv2
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader ,random_split,Subset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection, Accuracy
import torch.nn.functional as F
import torch
import os
#import albumentations as A
from CNN_activations import Activation_Function

class Lightning_CNN(L.LightningModule):
    def __init__(self,layers,kernel_size,pool_kernel,pool_stride,dense_layer_size,batch_normalization,drop_out,a_fun,optimizer,dense_layer_output,learning_rate):
        super().__init__()
        self.batch_normalization=batch_normalization
        self.drop_out=drop_out
        self.optimizer=optimizer
        act_object=Activation_Function()
        self.dense_layer_output=dense_layer_output
        self.act_fun=act_object.activation_Function(a_fun)
        self.learning_rate=learning_rate
        self.conv1 = nn.Conv2d(3,layers[0], kernel_size=kernel_size[0], padding=1)
        self.b1=nn.BatchNorm2d(layers[0])
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel[0], stride=pool_stride[0])
        self.conv2 = nn.Conv2d(layers[0],layers[1],kernel_size=kernel_size[1], padding=1)
        self.b2=nn.BatchNorm2d(layers[1])
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel[1], stride=pool_stride[1])
        self.conv3 = nn.Conv2d(layers[1],layers[2], kernel_size=kernel_size[2], padding=1)
        self.b3=nn.BatchNorm2d(layers[2])
        self.pool3 = nn.MaxPool2d(kernel_size=pool_kernel[2], stride=pool_stride[2])
        self.conv4 = nn.Conv2d(layers[2],layers[3], kernel_size=kernel_size[3], padding=1)
        self.b4=nn.BatchNorm2d(layers[3])
        self.pool4 = nn.MaxPool2d(kernel_size=pool_kernel[3], stride=pool_stride[3])
        self.conv5 = nn.Conv2d(layers[3],layers[4], kernel_size=kernel_size[4], padding=1)
        self.b5=nn.BatchNorm2d(layers[4])
        self.pool5 = nn.MaxPool2d(kernel_size=pool_kernel[4], stride=pool_stride[4])
        self.dropout = nn.Dropout(p=drop_out)
        self.fc1 = nn.Linear(dense_layer_size, self.dense_layer_output)
        self.fc2 = nn.Linear(self.dense_layer_output, 10)
    def forward(self,x):
        if self.batch_normalization==True:
            x=self.pool1(self.act_fun(self.b1(self.conv1(x))))
            x=self.pool2(self.act_fun(self.b2(self.conv2(x))))
            x=self.pool3(self.act_fun(self.b3(self.conv3(x))))
            x=self.pool4(self.act_fun(self.b4(self.conv4(x))))
            x=self.pool5(self.act_fun(self.b5(self.conv5(x))))
        else:
            x=self.pool1(self.act_fun(self.conv1(x)))
            x=self.pool2(self.act_fun(self.conv2(x)))
            x=self.pool3(self.act_fun(self.conv3(x)))
            x=self.pool4(self.act_fun(self.conv4(x)))
            x=self.pool5(self.act_fun(self.conv5(x)))

        x=self.dropout(x)
        x=torch.flatten(x,1)
        x=self.dropout(x)
        x=self.act_fun(self.fc1(x))
        x=self.fc2(x)
        return x
    def training_step(self, batch, batch_idx):
        inputs,labels=batch
        output=self(inputs)
        _,preds = torch.max(output, dim=1)
        loss=F.cross_entropy(output,labels)
        #train_acc = torch.mean(preds == labels)
        #print(pred.shape)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    def configure_optimizers(self):
        if self.optimizer=='adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer
        if self.optimizer=='nadam':
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate)
            return optimizer
        if self.optimizer=='sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
            return optimizer

    def validation_step(self,batch,batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        val_loss = F.cross_entropy(y_pred, y)

        # Compute validation accuracy
        _, predicted = torch.max(y_pred, 1)
        val_acc = torch.sum(predicted == y).item() / y.size(0)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return val_loss


    def test_step(self, batch, batch_idx):
        x,y=batch
        pred=self(x)
        loss=F.cross_entropy(pred,y)
        _, predicted = torch.max(pred, 1)
        accuracy = torch.sum(predicted == y).item() / y.size(0)
        #print(predicted,accuracy)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss}