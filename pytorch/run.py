from typing import Any
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.experimental.callback import Callback

class DataCallback(Callback):
    def __init__(self) -> None:
        pass

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        pass

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()