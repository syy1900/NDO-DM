import torch
import logging



def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

#  beta schedule
def sigmoid_beta_schedule(timesteps, beta_min=1e-4, beta_max=0.02, steepness=12.0):
    t = torch.arange(1, timesteps + 1, dtype=torch.float32)
    T = timesteps
    x = steepness * (t - 1) / (T - 1) - steepness / 2
    sigmoid = torch.sigmoid(x)
    return sigmoid * (beta_max - beta_min) + beta_min

def get_beta_schedule(name, timesteps, beta_min, beta_max, **kwargs):
    if name == "linear":
        logging.info("linear")
        return torch.linspace(beta_min, beta_max, timesteps)
    elif name == "sigmoid":
        logging.info("sigmoid")
        steepness = kwargs.get("steepness", 12.0)
        return sigmoid_beta_schedule(timesteps, beta_min, beta_max, steepness)
    else:
        raise ValueError(f"Unknown beta schedule: {name}")
