import numpy as np
import pandas as pd 
import pytorch_lightning as L
from torchvision import transforms, models,datasets
#import cv2
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader ,random_split,Subset
import matplotlib.pyplot as plt 
import torchvision.models as models
import torch.nn as nn 
import torch.optim as optim 
from torchmetrics import MetricCollection, Accuracy
import torch.nn.functional as F
import torch
#from albumentations.pytorch.transforms import ToTensorV2
import wandb
# Define a LightningModule class for pretrained CNN
class lightning_pretrained_CNN(L.LightningModule):
    def __init__(self,model_name,unfreeze_layers,optimizer,learning_rate):
        super().__init__()
        self.learning_rate=learning_rate
        self.optimizer=optimizer
        self.model_name=model_name
        # Load pre-trained model based on provided model_name
        if self.model_name=='ResNet50':
            self.model=models.resnet50(pretrained=True)
            #print('hi')
        if self.model_name=='GoogLeNet':
            self.model=models.googlenet(pretrained=True)
        if self.model_name=='InceptionV3':
            self.model=models.inception_v3(pretrained=True,transform_input=True)
        freeze_index=0
        for param in self.model.parameters():      # Freeze layers except for the last few layers specified by unfreeze_layers
            if freeze_index< (len(list(self.model.parameters())) - (unfreeze_layers + 2)):
                param.requires_grad = False
            else:
                break
            freeze_index=freeze_index+1
        num_feature=self.model.fc.in_features
        self.model.fc=nn.Linear(num_feature,10)       # Modify the fully connected layer to adapt to the number of classes to 10 class classification
    def forward(self, x):       # Forward pass of the model
        return self.model(x)

    def training_step(self, batch, batch_idx):    # Training step
        inputs,labels=batch
        output=self.forward(inputs)
        #print(type(output.logits))
        #print(output.shape)
        #print(output.logits)
        if self.model_name=='InceptionV3':
            _,preds = torch.max(output.logits, dim=1)
            loss=F.cross_entropy(output.logits,labels)   # Calculate loss
        #train_acc = torch.mean(preds == labels)
        #print(pred.shape)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)   # Log training loss
        else:
            _,preds = torch.max(output, dim=1)
            loss=F.cross_entropy(output,labels)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    def configure_optimizers(self):       # Configure optimizer based on arguments
        if self.optimizer=='adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer
        if self.optimizer=='nadam':
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate)
            return optimizer
        if self.optimizer=='sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
            return optimizer

    def validation_step(self,batch,batch_idx):    # Validation step
        x, y = batch
        y_pred = self.forward(x)
        val_loss = F.cross_entropy(y_pred, y)        # Calculate validation loss

        # Compute validation accuracy
        _, predicted = torch.max(y_pred, 1)
        val_acc = torch.sum(predicted == y).item() / y.size(0)
        # Log validation loss and accuracy
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return val_loss


    def test_step(self, batch, batch_idx):
        x,y=batch
        pred=self.forward(x)
        loss=F.cross_entropy(pred,y)    # Calculate test loss
        _, predicted = torch.max(pred.data, 1)
        accuracy = torch.sum(predicted == y).item() / y.size(0)
        #print(predicted,accuracy)      # Calculate test accuracy
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)     # Log test loss and accuracy
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss}
