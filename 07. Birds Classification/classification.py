import os
import cv2
import numpy as np
import albumentations as A

from itertools import chain

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from skimage.io import imread

import lightning as L

SIZE = 300
RGB_MEAN = (125.005, 130.891, 119.178)
RGB_STD = (46.192, 46.081, 48.974)


# ========= Birds Dataset =========
class BirdsDataset(data.Dataset):
    def __init__(
            self,
            mode: str,
            gt: dict,
            img_dir: str,
            train_fraction: float = 0.8,
            transform = None,
            normalize: bool = True,
    ):
        self._items = []
        self._transform = transform
        self._normalize = normalize

        inv_gt = {}
        for k, v in gt.items():
            inv_gt[v] = inv_gt.get(v, []) + [k]

        split = int(train_fraction * len(inv_gt[next(iter(inv_gt))]))

        if mode == "train":
            img_names = chain(*[imgs[:split] for imgs in inv_gt.values()])
        elif mode == "valid":
            img_names = chain(*[imgs[split:] for imgs in inv_gt.values()])
        elif mode == "sample":
            img_names = chain(*[imgs[:2] for imgs in inv_gt.values()])
        else:
            raise RuntimeError(f"Invalide mode {mode!r}")

        for img in img_names:
            self._items.append((os.path.join(img_dir, img), gt[img]))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path, label = self._items[index]
        
        img = imread(img_path)
        img = np.array(img).astype(np.float32)

        x, y = img.shape
        if x < y: 
            x, y = SIZE, int(SIZE * y / x)
        else:
            x, y = int(SIZE * x / y), SIZE

        img = cv2.resize(img, (x, y))
        img = A.CenterCrop(width=SIZE, height=SIZE)(image=img)["image"]

        if len(img.shape) == 2:
            img = np.stack([img]*3)

        if self._transform:
            img = self._transform(image=img)["image"]

        if self._normalize:
            img = A.Normalize(mean=RGB_MEAN, std=RGB_STD)(image=img)["image"]                  
        image = torch.from_numpy(img.transpose(2, 0, 1)).float()

        return image, label
    


# ========= MODEL =========
class EfficientNetB3Classifier(L.LightningModule):
    def __init__(self, num_classes, pretrained=False, train_last_n_layers=1):
        super().__init__()

        # self.mode = models.efficient_b3(pretrained=pretrained)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # Unfreeze all layers
        for child in list(self.model.children()):
            for p in child.parameters():
                p.requires_grad = True

        # Freeze all except last N
        for child in list(self.model.children())[:train_last_n_layers]:
            for p in child.parameters():
                p.requires_grad = False

    def forward(self, x):
        return nn.LogSoftmax(self.model(x), dim=1)            


class BirdsModel(L.LightningModule):
    def __init__(self, pretrained=False, train_last_n_layers=1):
        super().__init__()
        self.model = EfficientNetB3Classifier(50, pretrained, train_last_n_layers)
        self.train_loss = []
        self.valid_acc = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),  lr=1e-4)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.96,
            verbose=True,
        )    

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 3,
            "monitor": "valid_acc",
        }

        return [optimizer], [lr_scheduler_config]
    
    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "valid")
    
    def _step(self, batch, kind):
        x, y = batch
        p = self.model(x)
        loss = F.nll_loss(p, x)
        accs = torch.sum(p.argmax(dim=1) == y) / y.shape[0]

        self.train_loss.append(loss.detach())
        self.valid_acc.append(accs)

        metrics = {
            f"{kind}_accs": accs,
            f"{kind}_loss": loss,
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            logger=True,
            on_step=kind == "train",
            on_epoch=True,
        )

        return loss
    
    def on_train_epoch_end(self):
        epoch_loss = torch.stack(self.train_loss).mean()
        print(
            f"Epoch {self.trainer.current_epoch},",
            f"train_loss: {epoch_loss.item():.3f}",
        )
        self.train_loss.clear()

    def on_validation_epoch_end(self):
        epoch_accs = torch.stack(self.valid_acc).float().mean()
        print(
            f"Epoch {self.trainer.current_epoch},",
            f"valid_accs: {epoch_accs.item():.3f}",
        )
        self.valid_acc.clear()



def train_classifier(train_gt: dict,
                     train_img_dir: str,
                     fast_train=True):
    
    Transform = A.Compose([
        A.Rotate(limit=70, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    ds_train = BirdsDataset(
        mode="train",
        gt=train_gt,
        img_dir=train_img_dir,
        transform=Transform
    )

    ds_valid = BirdsDataset(
        mode="valid",
        gt=train_gt,
        img_dir=train_img_dir
    )

    ds_sample = BirdsDataset(
        mode="sample",
        gt=train_gt,
        img_dir=train_img_dir
    )

    dl_train = data.DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    dl_valid = data.DataLoader(
        ds_valid,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    dl_sample = data.DataLoader(
        ds_sample,
        batch_size=1, 
        shuffle=False,
        num_workers=1
    )

    model = BirdsModel(pretrained=not fast_train, train_last_n_layers=3)

    if fast_train:
        trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=1,
            devices=1,
            enable_checkpointing=False
        )
        trainer.fit(model, dl_train, dl_sample, dl_sample)
    else:
        checkpoint = L.pytorch.callback.ModelCheckpoint(
            dirpath="l_model",
            filename="{epoch}-{valid_accs:.3f}",
            monitor="valid_accs",
            mode="max",
            save_top_k=1,
            save_last=True
        )

        earlystopping = L.pytorch.callbacks.EarlyStopping(
            monitor="valid_accs",
            mode="max",
            patience=4,
            verbose=True
        )
        trainer = L.Trainer(
            accelerator="gpu",
            max_epochs=100,
            devices=1,
            enable_checkpointing=True,
            callbacks=[checkpoint, earlystopping]
        )    
        trainer.fit(model, dl_train, dl_valid)
        torch.save(model.to("cpu").state_dict(), "birds_model.pt")

    return model


def classify(model_path: str, test_img_dir: str):
    model = BirdsModel(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    img_names = sorted(os.listdir(test_img_dir))
    predictions = {}

    for img_name in img_names:
        img_path = os.path.join(test_img_dir, img_name)
        image = imread(img_path)
        image = np.array(image).astype(np.float32)

        x, y = image.shape
        if x < y:
            x, y = SIZE, int(SIZE * x / y)
        else:
            x, y = int(SIZE * y / x), SIZE
        image = cv2.resize(image, (x, y))
        image = A.CenterCrop(width=SIZE, height=SIZE)(image=image)["image"]

        if len(image.shape) == 2:
            image = np.dstack([image]*3)

        image = A.Normalize(mean=RGB_MEAN, std=RGB_STD)(image=image)["image"]
        image = torch.from_numpy(image.transpose(2, 0, 1))

        pred = model(image.unsqueeze(0)).argmax(dim=1).numpy()[0]
        predictions.update({img_name: pred})

    return predictions    
