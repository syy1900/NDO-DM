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


def train_ddpm(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    *,
    workdir,
    config,
):
    # Preparation
    savedir = workdir / "checkpoints"
    savedir.mkdir(exist_ok=True)

    samples_dir = workdir / "samples"
    samples_dir.mkdir(exist_ok=True)

    device = next(model.parameters()).device
    trainer = VanillaDDPMTrainer(config).to(device)
    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=config.train.ema_rate,
    )

    # Training step
    def train_step():
        model.train()
        metrics = {"loss": 0}

        for X, y in tqdm(dataloader):
            X = X.to(device)
            if config.data.class_condition:
                y = y.to(device) + 1
            else:
                y = torch.zeros_like(y, dtype=torch.long).to(device)
            loss = trainer(model, X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metrics["loss"] += loss.item()
            ema.update(model.parameters())

        for key in metrics:
            metrics[key] /= len(dataloader)

        return metrics

    # Test step
    def test_step(epoch):
        model.eval()
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        images = sample_example_image(model, config)
        save_image(images, samples_dir / f"{epoch + 1}.png")
        ema.restore(model.parameters())

    def save_step(epoch):
        torch.save(model.module.state_dict(),
                   savedir / f"model_{epoch+1}.ckpt")
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        torch.save(model.module.state_dict(), savedir / f"ema_{epoch+1}.ckpt")
        ema.restore(model.parameters())

    save_step(-1)
    test_step(-1)

    # Training loop
    for epoch in range(config.train.epochs):
        metrics = train_step()
        logging.info(f"Epoch{epoch+1}: {metrics}")
        with torch.no_grad():
            if (epoch + 1) % 10 == 0:
                save_step(epoch)
            test_step(epoch)



def train_dp_promise(
    model: torch.nn.Module,
    dataloader1,
    dataloader2,
    *,
    config,
    workdir,
):
    savedir = workdir / "checkpoints"
    savedir.mkdir(exist_ok=True)

    samples_dir = workdir / "samples"
    samples_dir.mkdir(exist_ok=True)

    privacy_engine = PrivacyEngine()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr1)

    device = next(model.parameters()).device
    trainer = DPPromiseTrainer(config).to(device)

    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=config.train.ema_rate,
    )

    # Phase I
    dataloader1 = DPDataLoader.from_data_loader(dataloader1)

    def train_non_private():
        all_loss = 0
        step = 0
        model.train()
        for X, y in tqdm(dataloader1):
            X = X.to(device)
            y = torch.zeros_like(y, dtype=torch.long).to(device)

            loss = trainer(model, X, y, phase="1")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ema.update(model.parameters())

            all_loss += loss.item()
            step += 1

        all_loss /= step
        logging.info(f"Epoch{epoch+1}: loss1 {all_loss}")

    logging.info("Training Phase I...")

    # Phase I training loop
    for epoch in range(config.train.epochs1):
        train_non_private()

    ema.copy_to(model.parameters())

    # Phase II
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr2)
    ema = ExponentialMovingAverage(
        model.parameters(),
        decay=config.train.ema_rate,
    )

    # config.diffusion.schedule = "linear"

    model, optimizer, dataloader2 = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader2,
        max_grad_norm=config.dp.max_grad_norm,
        noise_multiplier=config.dp.sigma,
        noise_multiplicity=config.train.num_noise_sample,
    )

    def train_private(epoch):
        all_loss = 0
        step = 0
        model.train()
        with BatchMemoryManager(
                data_loader=dataloader2,
                max_physical_batch_size=config.train.max_physical_batch_size,
                optimizer=optimizer) as memory_safe_dataloader:
            for X, y in tqdm(memory_safe_dataloader):
                X = X.to(device)
                if config.data.class_condition:
                    y = y.to(device) + 1
                else:
                    y = torch.zeros_like(y, dtype=torch.long).to(device)
                loss = trainer(model, X, y, phase="2")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if not optimizer._is_last_step_skipped:
                    ema.update(model.parameters())

                all_loss += loss.item()
                step += 1
        all_loss /= step
        logging.info(f"Epoch{epoch+1}: loss1 {all_loss}")
        # logging.info("S=",S_update)

        return all_loss

    def save_step(epoch):
        # if  epoch + 1 != config.train.epochs2:
        #     return
        if (epoch + 1) % 10 != 0 or epoch + 1 != config.train.epochs2:
            return
        torch.save(
            model._module.state_dict(),
            savedir / f"model_{epoch + 1}.ckpt",
        )
        torch.save(
            model._module.state_dict(),
            savedir / f"model.ckpt",
        )
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        torch.save(
            model._module.state_dict(),
            savedir / f"ema_{epoch + 1}.ckpt",
        )
        torch.save(
            model._module.state_dict(),
            savedir / f"ema.ckpt",
        )
        ema.restore(model.parameters())

    def test_step(epoch):
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        image = sample_example_image(model, config)
        save_image(image, samples_dir / f"{epoch + 1}.png")
        ema.restore(model.parameters())

    logging.info("Training Phase II...")

    # # Phase II training loop
    # for epoch in range(config.train.epochs2):
    #     train_private(epoch)
    #     with torch.no_grad():
    #         test_step(epoch)
    #         save_step(epoch)

    
    S_start=config.dp.S
    best_loss = float("inf")
    best_epoch = -1
    
    # Loss tracking for early stopping S update
    loss_history = []
    stop_S_update = False
    loss_change_threshold = 1e-5  # Threshold for loss change

    # S_start = 880
    # S_target = 900
    # warmup_epochs = 5
    # Phase II training loop
    from src.timestep import eps_from_config,get_sigma_from_config

    for epoch in range(config.train.epochs2): 
        r_e=epoch/config.train.epochs2
        if r_e > 0.5:
            r_e = 0.5
        

        # growth_speed = 2  # 
        # r_e = epoch / (config.train.epochs2 * growth_speed)
        # S_update = S_start + (config.diffusion.timesteps - S_start) * r_e

        S_update=S_start+(config.diffusion.timesteps-config.dp.S)*r_e
        # if S_update >= 925 :
        #     S_update=925
        config.dp.S = int(S_update)
        print(S_update)
        logging.info(f"S = {S_update}")




        new_sigma = get_sigma_from_config(config)

        
        new_sigma=float(round(new_sigma, 1))
        config.dp.sigma = new_sigma
        privacy_engine.noise_multiplier = new_sigma
        print(f"[Epoch {epoch}] sigma  {config.dp.sigma}")

           
        loss_val = train_private(epoch)  
        
        # Track loss and check for early stopping of S update
        loss_history.append(loss_val)
        if len(loss_history) >= 3:
            # Calculate loss changes between consecutive epochs
            change1 = abs(loss_history[-1] - loss_history[-2])
            change2 = abs(loss_history[-2] - loss_history[-3])
            if change1 < loss_change_threshold and change2 < loss_change_threshold:
                stop_S_update = True
                logging.info(f"Stopping S update: loss change below threshold for 3 consecutive epochs")
        
        with torch.no_grad():
            test_step(epoch)
        
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch + 1
            torch.save(model._module.state_dict(), savedir / f"best.ckpt")

            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            torch.save(model._module.state_dict(), savedir / f"ema_best.ckpt")
            ema.restore(model.parameters())

            logging.info(f"[BEST] Epoch {best_epoch} | Loss = {best_loss:.6f}")

        if epoch + 1 == config.train.epochs2:
            torch.save(model._module.state_dict(), savedir / f"last.ckpt")

            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            torch.save(model._module.state_dict(), savedir / f"ema_last.ckpt")
            ema.restore(model.parameters())


        # train_private(epoch)
        # with torch.no_grad():
        #     test_step(epoch)
        #     save_step(epoch)

        eps_from_config(config)

        


    logging.info("Training complete")

    def final_step():
        images, labels = sample_images(model, config, num_samples=60000)

        (workdir / "evaluation").mkdir(exist_ok=True)
        synthesis_name = f"synthesis.npz"
        np.savez(workdir / "evaluation" / synthesis_name,
                 data=images.numpy(), labels=labels.numpy())

    logging.info("Saving synthesis...")
    final_step()




# def train_dp_promise(
#     model: torch.nn.Module,
#     dataloader1,
#     dataloader2,
#     *,
#     config,
#     workdir,
# ):
#     savedir = workdir / "checkpoints"
#     savedir.mkdir(exist_ok=True)

#     samples_dir = workdir / "samples"
#     samples_dir.mkdir(exist_ok=True)

#     privacy_engine = PrivacyEngine()

#     optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr1)

#     device = next(model.parameters()).device
#     trainer = DPPromiseTrainer(config).to(device)

#     ema = ExponentialMovingAverage(
#         model.parameters(),
#         decay=config.train.ema_rate,
#     )

#     # Phase I
#     dataloader1 = DPDataLoader.from_data_loader(dataloader1)

#     def train_non_private():
#         all_loss = 0
#         step = 0
#         model.train()
#         for X, y in tqdm(dataloader1):
#             X = X.to(device)
#             y = torch.zeros_like(y, dtype=torch.long).to(device)

#             loss = trainer(model, X, y, phase="1")

#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()

#             ema.update(model.parameters())

#             all_loss += loss.item()
#             step += 1

#         all_loss /= step
#         logging.info(f"Epoch{epoch+1}: loss1 {all_loss}")

#     logging.info("Training Phase I...")

#     # Phase I training loop
#     for epoch in range(config.train.epochs1):
#         train_non_private()

#     ema.copy_to(model.parameters())

#     # Phase II
#     optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr2)
#     ema = ExponentialMovingAverage(
#         model.parameters(),
#         decay=config.train.ema_rate,
#     )

#     model, optimizer, dataloader2 = privacy_engine.make_private(
#         module=model,
#         optimizer=optimizer,
#         data_loader=dataloader2,
#         max_grad_norm=config.dp.max_grad_norm,
#         noise_multiplier=config.dp.sigma,
#         noise_multiplicity=config.train.num_noise_sample,
#     )

#     def train_private(epoch):
#         all_loss = 0
#         step = 0
#         model.train()
#         with BatchMemoryManager(
#                 data_loader=dataloader2,
#                 max_physical_batch_size=config.train.max_physical_batch_size,
#                 optimizer=optimizer) as memory_safe_dataloader:
#             for X, y in tqdm(memory_safe_dataloader):
#                 X = X.to(device)
#                 if config.data.class_condition:
#                     y = y.to(device) + 1
#                 else:
#                     y = torch.zeros_like(y, dtype=torch.long).to(device)
#                 loss = trainer(model, X, y, phase="2")

#                 optimizer.zero_grad(set_to_none=True)
#                 loss.backward()
#                 optimizer.step()

#                 if not optimizer._is_last_step_skipped:
#                     ema.update(model.parameters())

#                 all_loss += loss.item()
#                 step += 1
#         all_loss /= step
#         logging.info(f"Epoch{epoch+1}: loss1 {all_loss}")

#     def save_step(epoch):
#         if  epoch + 1 != config.train.epochs2:
#             return
#         torch.save(
#             model._module.state_dict(),
#             savedir / f"model_{epoch + 1}.ckpt",
#         )
#         torch.save(
#             model._module.state_dict(),
#             savedir / f"model.ckpt",
#         )
#         ema.store(model.parameters())
#         ema.copy_to(model.parameters())
#         torch.save(
#             model._module.state_dict(),
#             savedir / f"ema_{epoch + 1}.ckpt",
#         )
#         torch.save(
#             model._module.state_dict(),
#             savedir / f"ema.ckpt",
#         )
#         ema.restore(model.parameters())

#     def test_step(epoch):
#         ema.store(model.parameters())
#         ema.copy_to(model.parameters())
#         image = sample_example_image(model, config)
#         save_image(image, samples_dir / f"{epoch + 1}.png")
#         ema.restore(model.parameters())

#     logging.info("Training Phase II...")

#     # Phase II training loop
#     for epoch in range(config.train.epochs2):
#         train_private(epoch)
#         with torch.no_grad():
#             test_step(epoch)
#             save_step(epoch)

#     logging.info("Training complete")

#     def final_step():
#         images, labels = sample_images(model, config, num_samples=60000)

#         (workdir / "evaluation").mkdir(exist_ok=True)
#         synthesis_name = f"synthesis.npz"
#         np.savez(workdir / "evaluation" / synthesis_name,
#                  data=images.numpy(), labels=labels.numpy())

#     logging.info("Saving synthesis...")
#     final_step()