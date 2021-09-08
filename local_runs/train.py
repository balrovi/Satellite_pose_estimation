import argparse
import os


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils import PyTorchSatellitePoseEstimationDataset
from submission import SubmissionWriter
# from pytorch_lightning.plugins import DDPPlugin

# output will be logged, separate output from previous log entries.
print('-'*100)


def add_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser

class SatellitePoseEstimationModel(pl.LightningModule):
    def __init__(self, submission = None) :
        super().__init__()
        initialized_model = models.resnet18(pretrained=True)
        num_ftrs = initialized_model.fc.in_features
        initialized_model.fc = torch.nn.Linear(num_ftrs, 7)
        self.model = initialized_model
        self.submission = submission

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch ,batch_idx):
        x,y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat.float(),y.float())
        self.log('step', self.trainer.current_epoch+1)
        self.log('losses', {'train': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat.float(),y.float())
        self.log('step', self.trainer.current_epoch+1)
        self.log('losses', {'valid': loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 0.001)

    def test_step(self, batch, batch_idx):
        inputs, filenames = batch
        outputs = self.model(inputs)

        q_batch = outputs[:, :4].cpu().numpy()
        r_batch = outputs[:, -3:].cpu().numpy()

        for filename, q, r in zip(filenames, q_batch, r_batch):
            self.submission.append_test(filename, q, r)

class DataModule(pl.LightningDataModule) :
    def __init__(self, batch_size = 32, num_workers = 8, speed_root=''):
        super().__init__()
        self.batch_size = batch_size
        #num_workers = 4*gpu_num
        self.num_workers = num_workers
        self.speed_root = speed_root

    def setup(self, stage = None):
        #Transforms
        data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        full_dataset = PyTorchSatellitePoseEstimationDataset('train', self.speed_root, data_transforms)
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset,
                                                                   [int(len(full_dataset) * .8),
                                                                    int(len(full_dataset) * .2)])
        if stage == "test" or stage is None:
            self.test_dataset = PyTorchSatellitePoseEstimationDataset('test', self.speed_root, data_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers = self.num_workers)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        dest='data_path',
                        default='data',
                        help='data folder mounting point')

    parser.add_argument("--logdir", default="./logs", type=str)

    parser.add_argument("--local", default=1, type=int)

    parser = add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    # parse the parameters passed to the this script
    args = parser.parse_args()

    if args.local :
        data_path = "../../speed"
    else :
        data_path = args.data_path

    trial_name = f"first_model_{args.max_epochs}epochs"

#     MySubmission = SubmissionWriter()

    model = SatellitePoseEstimationModel()
    dm = DataModule(batch_size = args.batch_size, num_workers = args.num_workers, speed_root = data_path )

    tb_logger = TensorBoardLogger(args.logdir, name = trial_name)


    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger) #, plugins=DDPPlugin(find_unused_parameters=False))
    try :
        trainer.fit(model, dm)
    except :
        print("ERROR : The model stoped training !")
    finally :
        print('Saving model...')
        trainer.save_checkpoint(f"outputs/{trial_name}.ckpt")
#         trainer.test(model = model, datamodule = dm)
#         print(MySubmission.test_results)
#         MySubmission.export(out_dir="./outputs", suffix= trial_name)
    print('Done!')
    print('-'*100)
