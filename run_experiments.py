import os
import hydra
import mlflow
from omegaconf import DictConfig
from log import *
from train import *
from utils.utils import *
from model.simpleFCNN import *


@hydra.main(config_name="config", version_base=None, config_path="config")
def rum_experiments(cfg: DictConfig) -> None:
    # ワーキングディレクトリの設定
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)

    # MLFLOW_TRACKING_URIを設定
    tracking_dir = os.path.join(script_directory, cfg.mlflow.tracking_dir)
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_uri = f"file:///{tracking_dir.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(tracking_uri)
    
    # MLFLOW開始設定
    if cfg.mlflow.enable:
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        mlflow.start_run()
        # save_userparams_on_mlflow(cfg)
    
    # Log設定、乱数固定、学習
    manager = LoggerManager(enable_mlflow=cfg.mlflow.enable)
    manager.run(DeviceLogger(cfg))
    manager.run(UserParamLogger(cfg))
    set_seed(42)
    model = train(cfg, manager)

    # MLFLOW終了設定
    if cfg.mlflow.enable:
        mlflow.end_run()
        

if __name__ == '__main__':
    rum_experiments()
