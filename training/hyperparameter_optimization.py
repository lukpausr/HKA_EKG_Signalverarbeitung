import os
from pyexpat import model
import sys
import datetime

import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from data.datamodule import ECG_DataModule
from model.model import UNET_1D

class OptunaTrainer:
    def __init__(self, model, config):
        self.config = config
        self.datamodule = ECG_DataModule
        self.model = model

    def _build_transform(self):
        return None

    def _setup_wandb_logger(self):

        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        print(self.config['data_cols'])
        lead_str = '-'.join(self.config['data_cols'])
        print(lead_str)
        exp_name = (
            f"{self.config['model_name']}"
            f"_bs{self.config['batch_size']}"
            f"_lr{self.config['learning_rate']:.1e}"    # Exponential format
            f"_wd{self.config['weight_decay']:.1e}"
            f"_opt{self.config['optimizer']}"
            f"_sch{self.config['scheduler'] if self.config['scheduler'] else 'None'}"
            f"_acc{self.config['accumulate_grad_batches']}"
            f"_leads{lead_str}"
            f"_{now_str}"
        )
        self.config['experiment_name'] = exp_name

        return WandbLogger(
            project=self.config['wandb_project_name'],
            name=exp_name,
            config={
                'sweep_id': self.config['sweep_id'],
                'batch_size': self.config['hpo_batch_size'],
                'sampling_rate': self.config['sampling_rate'],
                'max_epochs': self.config['max_epochs'],
                'accumulate_grad_batches': self.config['accumulate_grad_batches'],
                'precision': self.config['precision'],
                'optimizer': self.config['optimizer'],
                'learning_rate': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay'],
                'scheduler': self.config['scheduler'],
                'model_name': self.config['model_name'],
                'dataset_name': self.config['dataset_name'],
                'leads': self.config['data_cols']
            }
        )
    
    def suggest_ecg_leads(self, trial, available_leads):

        mask = []
        for lead in available_leads:
            mask.append(trial.suggest_categorical(f"{lead}", [0, 1]))
        selected_leads = [lead for lead, m in zip(available_leads, mask) if m == 1]
        if sum(mask) == 0:
            selected_leads = [available_leads[0]]  # Ensure at least one lead is selected
        return selected_leads

    def run_training(self, trial):

        self.config['batch_size'] = trial.suggest_categorical('batch_size', self.config['hpo_batch_size'])
        self.config['max_epochs'] = trial.suggest_int('max_epochs', self.config['hpo_min_epochs'], self.config['hpo_max_epochs'], step=10)
        self.config['accumulate_grad_batches'] = trial.suggest_categorical('accumulate_grad_batches', self.config['hpo_accumulate_grad_batches'])
        self.config['precision'] = trial.suggest_categorical('precision', self.config['hpo_precision'])
        self.config['optimizer'] = trial.suggest_categorical('optimizer', self.config['hpo_optimizers'])
        self.config['learning_rate'] = trial.suggest_float('learning_rate', self.config['hpo_min_learning_rate'], self.config['hpo_max_learning_rate'], log=True)
        self.config['weight_decay'] = trial.suggest_float('weight_decay', self.config['hpo_min_weight_decay'], self.config['hpo_max_weight_decay'], log=True)
        self.config['scheduler'] = trial.suggest_categorical('scheduler', self.config['hpo_scheduler'])
        
        self.config['data_cols'] = self.suggest_ecg_leads(trial, self.config['available_leads'])
        print(f"Selected leads for this trial: {self.config['data_cols']}")

        transform = self._build_transform() # not utilized, but acts as a dummy for data augmentation if required

        # Setup wandb_logger for experiment tracking
        wandb_logger = self._setup_wandb_logger()

        # Setup data module
        dm = self.datamodule(
            data_dir=self.config['path_to_data'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            transform=transform,
            persistent_workers=self.config['persistent_workers'],
            feature_list=self.config['feature_list'],
            data_cols=self.config['data_cols']
        )

        # Initialize model
        self.model = self.model(
            in_channels=len(self.config['data_cols']),
            layer_n=512,
            out_channels=len(self.config['feature_list']),
            kernel_size=5,
            learning_rate=self.config['learning_rate'],
            optimizer_name=self.config['optimizer'],
            weight_decay=self.config['weight_decay'],
            scheduler_name=self.config['scheduler'],
            model_name=self.config['experiment_name']
        )

        # Setup trainer
        trainer = Trainer(
            max_epochs=self.config['max_epochs'],
            precision=self.config['precision'],
            accumulate_grad_batches=self.config['accumulate_grad_batches'],
            accelerator="auto",
            devices="auto",
            strategy="auto",
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
            logger=wandb_logger,
            enable_progress_bar=True,
            log_every_n_steps=10
        )

        # Train and handle exceptions, should any occur
        try:
            trainer.fit(model=self.model, datamodule=dm)
            checkpoint_path = os.path.join(self.config['path_to_models'], f"checkpoints/{self.config['experiment_name']}.ckpt")
            trainer.save_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"Error occurred during training: {e}")
            wandb.finish()
            return float('inf')

        val_loss = trainer.callback_metrics.get("val_loss") 
        wandb.finish()

        # Save the model's state_dict to the path specified in config
        # self.model.model_name = self.config['experiment_name']
        save_path = os.path.join(self.config['path_to_models'], self.config['experiment_name'], "model_best_ckpt.ckpt")
        trainer.save_checkpoint(save_path)
        print(f"Model checkpoint saved as {save_path}")

        print(f"Optimization finished with best validation loss: {val_loss.item() if val_loss else float('inf')}")
        return val_loss.item() if val_loss else float('inf')
    
    def run_test(self):

        transform = self._build_transform() # not utilized, but acts as a dummy for data augmentation if required

        # Setup data module
        dm = self.datamodule(
            data_dir=self.config['path_to_data'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            transform=transform,
            persistent_workers=self.config['persistent_workers'],
            feature_list=self.config['feature_list'],
            data_cols=self.config['data_cols']
        )

        trainer = Trainer()
        # dm.setup(stage="test")
        print(self.model.model_name)
        self.model.multi_tolerance_metrics.path_to_history_data = os.path.join(self.config['path_to_models'], self.config['experiment_name'])
        trainer.test(model=self.model, datamodule=dm)

