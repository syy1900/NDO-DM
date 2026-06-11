import argparse

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from src.schedules import linear_beta_schedule
from src.utils import load_dataset_from_config
from src.runners import train_dp_promise

from scipy import optimize
from scipy.stats import norm
from math import sqrt
import numpy as np


# Dual between mu-GDP and (epsilon,delta)-DP
def delta_eps_mu(eps, mu):
    return norm.cdf(-eps / mu +
                    mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):

    def f(x):
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def gdp_mech(sample_rate1, sample_rate2, niter1, niter2, sigma,
             alpha_cumprod_S, d, delta):
    mu_1 = sample_rate1 * sqrt(niter1 * (np.exp(4 * d * alpha_cumprod_S / (1 - alpha_cumprod_S)) - 1))
    mu_2 = sample_rate2 * sqrt(niter2 * (np.exp(1 / (sigma ** 2)) - 1))
    mu = sqrt(mu_1 ** 2 + mu_2 ** 2)  
    epsilon = eps_from_mu(mu, delta)  

    return epsilon



from scipy import optimize

def sigma_from_eps(epsilon_target, sample_rate1, sample_rate2, niter1, niter2,
                   alpha_cumprod_S, d, delta):

    def objective(sigma):
        epsilon = gdp_mech(
            sample_rate1=sample_rate1,
            sample_rate2=sample_rate2,
            niter1=niter1,
            niter2=niter2,
            sigma=sigma,
            alpha_cumprod_S=alpha_cumprod_S,
            d=d,
            delta=delta,
        )
        return epsilon - epsilon_target

    result = optimize.root_scalar(objective, bracket=[0.1, 50.0], method='brentq')
    if result.converged:
        return result.root
    else:
        raise ValueError("Failed to converge when solving for sigma")


def eps_from_config(config):
    dataset = load_dataset_from_config(config)
    d = config.data.img_ch * config.data.img_size * config.data.img_size

    dataloader1 = DataLoader(
        dataset,
        batch_size=config.train.batch_size1,
    )

    dataloader2 = DataLoader(
        dataset,
        batch_size=config.train.batch_size2,
    )

    prob1 = 1 / len(dataloader1)
    prob2 = 1 / len(dataloader2)
    niter1 = config.train.epochs1 * len(dataloader1)
    niter2 = config.train.epochs2 * len(dataloader2)

    betas = linear_beta_schedule(
        config.diffusion.timesteps,
        config.diffusion.beta_start,
        config.diffusion.beta_end,
    )

    alphas = 1 - betas 
    alphas_cumprod = torch.cumprod(alphas, axis=0) 

    alpha_cumprod_S = alphas_cumprod[config.dp.S - 1].numpy() 
    sigma_update = sigma_from_eps(
    epsilon_target=config.dp.epsilon,
    sample_rate1=prob1,
    sample_rate2=prob2,
    niter1=niter1,
    niter2=niter2,
    alpha_cumprod_S=alpha_cumprod_S,
    d=d,
    delta=config.dp.delta
)

    epsilon = gdp_mech(
        sample_rate1=prob1,
        sample_rate2=prob2,
        niter1=niter1,
        niter2=niter2,
        sigma=sigma_update,
        alpha_cumprod_S=alpha_cumprod_S,
        d=d,
        delta=config.dp.delta,
    )

    return epsilon






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    opt, _ = parser.parse_known_args()
    config = OmegaConf.load(opt.config)

    delta = config.dp.delta
    eps = eps_from_config(config)
    print(f"(epsilon, delta) = ({eps}, {delta})")




