import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchvision
from torchmetrics.functional import f1_score


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                      padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

        # 🔑 Residual gates (start at 0 ⇒ identity)
        self.gamma_c = nn.Parameter(torch.zeros(1))
        self.gamma_s = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, _, _ = x.shape

        # ----- Channel attention -----
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1).view(B, C))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1).view(B, C))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(B, C, 1, 1)
        # residual gating: x * (1 + gamma * att)
        x = x * (1 + self.gamma_c * channel_att)

        # ----- Spatial attention -----
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_pool, max_pool], dim=1)))
        x = x * (1 + self.gamma_s * spatial_att)

        return x


class Classifier(L.LightningModule):
    def __init__(self, config, label_encoder=None):
        super().__init__()
        self.val_loss_history = []
        self.val_preds, self.val_targets = [], []
        self.test_preds, self.test_targets = [], []
        self.config = config

        # ----- Loss -----
        self.loss_fcn = getattr(torch.nn, config["BASEMODEL"]["Loss_Function"])()
        if config["BASEMODEL"]["Loss_Function"] == "CrossEntropyLoss":
            w = (
                torch.tensor(config["DATA"].get("weights"), dtype=torch.float32)
                if "weights" in config["DATA"]
                else torch.ones(config["DATA"]["Num_of_Classes"], dtype=torch.float32)
            )
            self.loss_fcn = nn.CrossEntropyLoss(
                weight=w, label_smoothing=config["REGULARIZATION"]["Label_Smoothing"]
            )

        # ----- Backbone -----
        backbone_fn = getattr(torchvision.models, config["BASEMODEL"]["Backbone"])
        self.backbone = backbone_fn(weights="DEFAULT")
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, config["DATA"]["Num_of_Classes"])

        # ----- Mask encoders -----
        self.mask_encoder = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.encoder_4d = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # ----- Inject CBAM blocks after ResNet layers -----
        # self.cbam1 = CBAM(self.backbone.layer1[-1].conv2.out_channels)
        # self.cbam2 = CBAM(self.backbone.layer2[-1].conv2.out_channels)
        self.cbam3 = CBAM(self.backbone.layer3[-1].conv2.out_channels)
        self.cbam4 = CBAM(self.backbone.layer4[-1].conv2.out_channels)



        self.activation = getattr(nn, config["BASEMODEL"]["Activation"])()
        self.save_hyperparameters()

    def forward(self, data):
        x = self.backbone.conv1(data["img"])
        if self.config["BASEMODEL"]["Mask_Input"]:
            if self.config["BASEMODEL"]["Input_Type"] == "3_Channel":
                x = x + self.mask_encoder(data["msk"])
            elif self.config["BASEMODEL"]["Input_Type"] == "4_Channel":
                x = self.encoder_4d(torch.cat([data["img"], data["msk"]], dim=1))

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        # x = self.cbam1(x)

        x = self.backbone.layer2(x)
        # x = self.cbam2(x)

        x = self.backbone.layer3(x)
        x = self.cbam3(x)

        x = self.backbone.layer4(x)
        x = self.cbam4(x)

        x = torch.mean(torch.mean(x, dim=2), dim=2)
        x = self.backbone.fc(x)
        return self.activation(x)
    def on_validation_epoch_start(self):
        self.val_preds, self.val_targets = [], []

    def on_validation_epoch_end(self):
        if self.val_preds and self.val_targets:
            preds = torch.cat(self.val_preds)
            targets = torch.cat(self.val_targets)
            val_f1 = f1_score(preds, targets, task="binary")
            self.log("val_f1_score", val_f1, prog_bar=True)

        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            v = float(val_loss.cpu().item())
            self.val_loss_history.append(v)
            print(f"\n📊 Epoch {self.current_epoch}: val_loss = {v:.3f}")
            print("Full val_loss history:", [round(x, 3) for x in self.val_loss_history])

    def on_test_epoch_start(self):
        self.test_preds, self.test_targets = [], []

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
        test_f1 = f1_score(preds, targets, task="binary")
        self.log("test_f1", test_f1, prog_bar=True)
        print(f"\n🏁 Test F1 Score: {test_f1:.4f}")

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
        main, attn = [], []
        for n,p in self.named_parameters():
            if n.startswith(("cbam",)): attn.append(p)
            else: main.append(p)

        optimizer = torch.optim.AdamW(
            [{"params": main, "lr": self.config['OPTIMIZER']['lr']},
            {"params": attn, "lr": self.config['OPTIMIZER']['lr'] * 0.5}],
            eps=self.config['OPTIMIZER']['eps'],
            weight_decay=self.config['REGULARIZATION']['Weight_Decay']
        )

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config["SCHEDULER"].get("T_0", 10),
            T_mult=self.config["SCHEDULER"].get("T_mult", 1),
            eta_min=self.config["SCHEDULER"].get("eta_min", 1e-6),
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[5]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
