import lightning as L
import torch
import torchvision
from torchvision.transforms.functional import pil_to_tensor
from torch.nn.functional import softmax, interpolate
from torchmetrics.functional import accuracy, precision, recall, f1_score
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_fscore_support, roc_auc_score, accuracy_score
from PIL import Image

class Classifier(L.LightningModule):
    def __init__(self, config, label_encoder=None):
        super().__init__()
        self.val_loss_history = []
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []
        self.config = config
        self.loss_fcn = getattr(torch.nn, self.config["BASEMODEL"]["Loss_Function"])()
        if self.config['BASEMODEL']['Loss_Function'] == 'CrossEntropyLoss':
            if "weights" in self.config['DATA']:
                w = torch.tensor(self.config['DATA']['weights'], dtype=torch.float32)
            else:
                w = torch.ones(self.config['DATA']['Num_of_Classes'], dtype=torch.float32)
            self.loss_fcn = torch.nn.CrossEntropyLoss(weight=w,
                                                      label_smoothing=self.config['REGULARIZATION']['Label_Smoothing'])

        self.LabelEncoder = label_encoder
        self.activation = getattr(torch.nn, self.config["BASEMODEL"]["Activation"])()
        backbone = getattr(torchvision.models, self.config['BASEMODEL']['Backbone'])
        self.backbone = backbone(weights='DEFAULT')
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, self.config['DATA']['Num_of_Classes'])
        self.mask_encoder = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
        #self.mask_encoder.bias.data = torch.ones(self.mask_encoder.bias.data.shape)
        #self.mask_encoder.weight.data = torch.ones(self.mask_encoder.weight.data.shape)
        self.encoder_4d = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.save_hyperparameters()

    def forward(self, data):
        x = self.backbone.conv1(data['img'])
        if self.config['BASEMODEL']['Mask_Input']:
            if self.config['BASEMODEL']['Input_Type'] == "3_Channel":
                x = x + self.mask_encoder(data['msk'])
            elif self.config['BASEMODEL']['Input_Type'] == "4_Channel":
                x = self.encoder_4d(torch.cat([data['img'], data['msk']], dim=1))

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        # x = x.squeeze()
        x = self.backbone.fc(x)
        x = self.activation(x)

        return x
    
    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_targets = []

    def on_validation_epoch_end(self):
        if self.val_preds and self.val_targets:
            preds = torch.cat(self.val_preds)
            targets = torch.cat(self.val_targets)
            val_f1 = f1_score(preds, targets, task="binary")
            self.log("val_f1_score", val_f1, prog_bar=True)

        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss_value = float(val_loss.cpu().item())
            self.val_loss_history.append(val_loss_value)
            print(f"\n📊 Epoch {self.current_epoch}: val_loss = {val_loss_value:.3f}")
            print("Full val_loss history:", [round(v, 3) for v in self.val_loss_history])

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
        
        test_f1 = f1_score(preds, targets, task="binary")
        self.log("test_f1", test_f1, prog_bar=True)
        print(f"\n🏁 Test F1 Score: {test_f1:.4f}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = softmax(self(batch), dim=1)
        return output, batch['coords'], batch['id']
        # return self.all_gather(output), self.all_gather(batch['coords']), self.all_gather(batch['id'])

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.loss_fcn(preds, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.loss_fcn(preds, target)
        self.log("val_loss", loss, prog_bar=True)

        self.val_preds.append(preds.argmax(dim=-1))
        self.val_targets.append(target)

        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.loss_fcn(preds, target)
        self.log("test_loss", loss)

        self.test_preds.append(preds.argmax(dim=-1))
        self.test_targets.append(target)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['OPTIMIZER']['lr'],
            eps=self.config['OPTIMIZER']['eps'],
            weight_decay=self.config['REGULARIZATION']['Weight_Decay']
        )

        # Warmup for first 5 epochs
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.3,   # start at 30% of initial lr
            total_iters=5       # for 5 epochs
        )

        # Cosine Annealing with Warm Restarts after warmup
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config['SCHEDULER'].get('T_0', 10),
            T_mult=self.config['SCHEDULER'].get('T_mult', 1),
            eta_min=self.config['SCHEDULER'].get('eta_min', 1e-6)
        )

        # Chain them: warmup first, then cosine restarts
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[5]   # switch after 5 warmup epochs
        )

        # Return both to Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",    # step per epoch
                "frequency": 1,
            }
        }


