import os
import sys
from omegaconf import OmegaConf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.utils import set_seed
from src.utils.log import LoggerManager
from src.training.train_model import train_model


if __name__ == '__main__':
    # ワーキングディレクトリの設定
    project_root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root_directory)
    print(project_root_directory)
    # 乱数設定、ログ設定  
    set_seed(42)
    # config読み込み
    cfg = OmegaConf.load("./config/config.yaml")
    # Log制御インスタンス
    manager = LoggerManager(enable_mlflow=False)
    # 学習
    train_model(cfg, manager)