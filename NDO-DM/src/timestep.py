import argparse

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
import logging

from src.schedules import linear_beta_schedule
from src.schedules import sigmoid_beta_schedule
from src.schedules import get_beta_schedule
from src.utils import load_dataset_from_config
from src.runners import train_dp_promise

from scipy import optimize
from scipy.stats import norm
from math import sqrt
import numpy as np

def delta_eps_mu(eps, mu):
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
    def f(eps):
        return delta_eps_mu(eps, mu) - delta

    f0 = f(0)
    f500 = f(500)
    print(f"[DEBUG] mu = {mu}, f(0) = {f0}, f(500) = {f500}")

    if f0 * f500 > 0:
        raise ValueError("Function does not change sign on [0, 500]. Try adjusting mu or delta.")

    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def compute_mu1(sample_rate, niter, alpha_cumprod_S, d):
    exp_term = 4 * d * alpha_cumprod_S / (1 - alpha_cumprod_S)
    exp_term = np.clip(exp_term, 0, 100)
    return sample_rate * sqrt(niter * (np.exp(exp_term) - 1))

def compute_mu2(sample_rate, niter, sigma):
    mu_t = sample_rate * sqrt(np.exp(1 / sigma ** 2) - 1)
    return sqrt(niter) * mu_t

def gdp_mech(sample_rate1, sample_rate2, niter1, niter2, sigma, alpha_cumprod_S, d, delta):
    mu_1 = compute_mu1(sample_rate1, niter1, alpha_cumprod_S, d)
    mu_2 = compute_mu2(sample_rate2, niter2, sigma)
    mu = sqrt(mu_1 ** 2 + mu_2 ** 2)
    return eps_from_mu(mu, delta)


from scipy import optimize

def sigma_from_eps(epsilon_target, sample_rate1, sample_rate2, niter1, niter2, delta,d,alpha_cumprod_S):
    def objective(sigma):
        try:
            epsilon = gdp_mech(
                sample_rate1=sample_rate1,
                sample_rate2=sample_rate2,
                niter1=niter1,
                niter2=niter2,
                sigma=sigma,
                d=d,
                alpha_cumprod_S=alpha_cumprod_S,
                delta=delta
            )
            print(f"[DEBUG] sigma = {sigma:.4f} -->  epsilon = {epsilon:.4f}")
            logging.debug(f"[DEBUG] sigma = {sigma:.4f} -->  epsilon = {epsilon:.4f}")
            return epsilon - epsilon_target
        except ValueError:
            return 1e6  # penalize non-convergence

    result = optimize.root_scalar(objective, bracket=[0.5, 50.0], method='brentq')
    if result.converged:
        return result.root
    else:
        raise ValueError("Failed to converge when solving for sigma")


def get_sigma_from_config(config):
    dataset = load_dataset_from_config(config)
    d = config.data.img_ch * config.data.img_size * config.data.img_size

    dataloader1 = DataLoader(dataset, batch_size=config.train.batch_size1)
    dataloader2 = DataLoader(dataset, batch_size=config.train.batch_size2)

    prob1 = 1 / len(dataloader1)
    prob2 = 1 / len(dataloader2)
    niter1 = config.train.epochs1 * len(dataloader1)
    niter2 = config.train.epochs2 * len(dataloader2)

    betas = get_beta_schedule(
        name=config.diffusion.schedule,
        timesteps=config.diffusion.timesteps,
        beta_min=config.diffusion.beta_start,
        beta_max=config.diffusion.beta_end,
        steepness=getattr(config.diffusion, "steepness", 12.0)
    )

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    S = max(1, config.dp.S)  # 
    alpha_cumprod_S = alphas_cumprod[S - 1].item()

    sigma = sigma_from_eps(
        epsilon_target=config.dp.epsilon,
        sample_rate1=prob1,
        sample_rate2=prob2,
        niter1=niter1,
        niter2=niter2,
        delta=config.dp.delta,
        d=d,
        alpha_cumprod_S=alpha_cumprod_S
    )

    logging.info(f"Updated sigma: {sigma:.4f} for S={S}")
    return sigma



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

    # betas = linear_beta_schedule(
    #     config.diffusion.timesteps,
    #     config.diffusion.beta_start,
    #     config.diffusion.beta_end,
    # )
    betas = get_beta_schedule(
        name=config.diffusion.schedule,
        timesteps=config.diffusion.timesteps,
        beta_min=config.diffusion.beta_start,
        beta_max=config.diffusion.beta_end,
        steepness=getattr(config.diffusion, "steepness", 12.0)
    )

    alphas = 1 - betas 
    alphas_cumprod = torch.cumprod(alphas, axis=0) 
    print('timestep-S:',config.dp.S)

    alpha_cumprod_S = alphas_cumprod[config.dp.S - 1].numpy() 


    sigma_update = sigma_from_eps(
    epsilon_target=config.dp.epsilon,
    sample_rate1=prob1,
    sample_rate2=prob2,
    niter1=niter1,
    niter2=niter2,
    delta=config.dp.delta,
    d=d,
    alpha_cumprod_S=alpha_cumprod_S
)
    
    print('sigma',sigma_update)

    epsilon = gdp_mech(
    sample_rate1=prob1,
    sample_rate2=prob2,
    niter1=niter1,
    niter2=niter2,
    sigma=sigma_update,alpha_cumprod_S=alpha_cumprod_S, d=d,
    delta=config.dp.delta
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




