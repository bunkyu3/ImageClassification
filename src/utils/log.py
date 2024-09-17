import os
import mlflow
import torch
from abc import ABC, abstractmethod
from omegaconf import OmegaConf


class BaseLogger(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def log_on_local(self):
        pass

    @abstractmethod
    def log_on_mlflow(self):
        pass


class MetricLogger(BaseLogger):
    def __init__(self, cfg, epoch, epoch_loss, accuracy=None):
        super().__init__(cfg)
        self.epoch = epoch
        self.epoch_loss = epoch_loss
        self.accuracy = accuracy

    def log_on_local(self):
        pass

    def log_on_mlflow(self):
        mlflow.log_metric("val_loss", self.epoch_loss, step=self.epoch)
        if self.accuracy is not None:
            mlflow.log_metric("accuracy", self.accuracy, step=self.epoch)


class ConfigLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def log_on_local(self):
        save_path = self.cfg.local.write_loc.config
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as file:
            OmegaConf.save(config=self.cfg, f=file.name)
        print(f"Saved Config to {save_path}")

    def log_on_mlflow(self):
        mlflow.log_artifact(self.cfg.local.write_loc.config, self.cfg.mlflow.write_loc.config)
        print(f"Saved Config on mlflow")


class BestModelLogger(BaseLogger):
    def __init__(self, cfg, model):
        super().__init__(cfg)
        self.model = model

    def log_on_local(self):
        save_path = self.cfg.local.write_loc.best_model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved Best Model to {save_path}")

    def log_on_mlflow(self):
        mlflow.log_artifact(self.cfg.local.write_loc.best_model, self.cfg.mlflow.write_loc.best_model)
        print(f"Saved Best Model on mlflow")


class DeviceLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        if torch.cuda.is_available():
            self.device = torch.cuda.get_device_name()
        else:
            self.device = "cpu"

    def log_on_local(self):
        pass
    
    def log_on_mlflow(self):
        mlflow.log_param("device", self.device)
        print(f"Saved Using Device on mlflow")


class UserParamLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def log_on_local(self):
        pass
    
    def log_on_mlflow(self):
        mlflow.log_params(self.cfg.hparam)
        print(f"Saved User parameter on mlflow")


class LoggerManager:
    def __init__(self, enable_mlflow):
        self.enable_mlflow = enable_mlflow

    def run(self, Logger: BaseLogger):
        Logger.log_on_local()
        if self.enable_mlflow:
            Logger.log_on_mlflow()