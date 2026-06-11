import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml
class DotDict(dict):
    
    def __init__(self, d=None):
        super().__init__()
        d = d or {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = DotDict(v)
            self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
from src.models import UNetModel          # 
from src.samplers import DDIMSampler      # 

from torchvision.utils import save_image
import torch
import numpy as np
import logging
from tqdm import tqdm

from src.trainers import DPPromiseTrainer
from src.trainers import VanillaDDPMTrainer
from src.ema import ExponentialMovingAverage
from src.utils import sample_example_image
from src.utils import sample_images

import importlib
opacus = importlib.import_module('src.opacus')

from opacus.data_loader import DPDataLoader
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine

def build_model(config):
    
    model = UNetModel(
        in_channels=config["data"]["img_ch"],
        model_channels=config["model"]["ch"],
        out_channels=config["data"]["img_ch"],
        num_res_blocks=config["model"]["num_res_blocks"],
        attention_resolutions=config["model"]["attn"],
        dropout=config["model"]["dropout"],
        channel_mult=tuple(config["model"]["ch_mult"]),
        num_classes=config["data"]["num_classes"] if config["data"]["class_condition"] else None,
    )
    return model


@torch.no_grad()
def sample_images(model, config, num_samples):
   
    device = next(model.parameters()).device
    sampler = DDIMSampler(config)

   
    if config["data"]["class_condition"]:
        labels = torch.randint(config["data"]["num_classes"], size=(num_samples,))
    else:
        labels = torch.zeros(size=(num_samples,), dtype=torch.long)

    dataloader = DataLoader(
        TensorDataset(labels),
        batch_size=config["sample"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    images = []
    all_labels = []
    for (y,) in tqdm(dataloader, desc="Sampling"):
        shape = (
            y.size(0),
            config["data"]["img_ch"],
            config["data"]["img_size"],
            config["data"]["img_size"],
        )
        y = y.to(device) + 1
        if not config["data"]["class_condition"]:
            y = torch.zeros_like(y, dtype=torch.long).to(device)

        X = sampler.sample(model, shape, y)
        X = (X + 1.) / 2.
        X = X.clamp(0., 1.)
        X = X.detach().cpu()

        images.append(X)
        all_labels.append(y.cpu())

    images = torch.cat(images)
    labels = torch.cat(all_labels)
    return images, labels


def main():
    
    workdir = Path("src/dp-promise_<job-id>_cifar10_eps5.0_0908-18:07:12")  
    config_path =  "src/config.yaml"
    ckpt_path = workdir / "checkpoints" / "last.ckpt"

    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

   
    config = DotDict(config_dict)  


    
    model = build_model(config).to("cuda")
    model.eval()

    
    state_dict = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state_dict, strict=False)

    
    num_samples = 60000   
    images, labels = sample_images(model, config, num_samples)

    
    eval_dir = workdir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    synthesis_path = eval_dir / "synthesis.npz"
    np.savez(synthesis_path, data=images.numpy(), labels=labels.numpy())

    print(f"[OK] Synthesis saved to {synthesis_path}")


if __name__ == "__main__":
    main()
