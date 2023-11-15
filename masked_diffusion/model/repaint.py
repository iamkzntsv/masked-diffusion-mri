# Copyright 2023 ETH Zurich Computer Vision Lab and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import torch
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RePaintPipeline:
    def __init__(self, unet, scheduler):
        super().__init__()

        self.unet = unet
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(self, image, mask_image, num_inference_steps, jump_length, jump_n_sample, device):
        self.num_inference_steps = num_inference_steps
        self.mask_image = mask_image
        self.jump_length = jump_length
        self.jump_n_sample = jump_n_sample
        self.device = device

        original_image = image.to(self.device)
        mask_image = mask_image.to(self.device)

        image = torch.randn(image.shape, device=self.device)

        self.scheduler.set_timesteps(
            self.num_inference_steps, self.jump_length, self.jump_n_sample, self.device
        )
        reverse_transform = get_reverse_transform(original_shape, transform_state)["image"]

        logger.info("Inpainting...")
        t_last = self.scheduler.timesteps[0] + 1
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
            if t < t_last:
                # predict the noise residual
                model_output = self.unet(image, t.unsqueeze(0))

                # compute previous image: x_t -> x_t-1
                image = self.scheduler.step(
                    model_output, t.squeeze(), image, original_image, mask_image
                ).prev_sample

            else:
                # compute the reverse: x_t-1 -> x_t
                image = self.scheduler.undo_step(image, t_last.squeeze())
            t_last = t

            if i % 100 == 0 or i == self.scheduler.timesteps-1:
                inpainted_image = reverse_transform(inpainted_image)
                inpainted_image = np.split(inpainted_image, args.batch_size, axis=0)
                wandb.log({"image_grid": wandb.Image(image)})

        return image


class RePaintScheduler:
    def __init__(
        self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02
    ):
        self.num_train_timesteps = num_train_timesteps
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)

        # Standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # Setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def set_timesteps(
        self, num_inference_steps: int, jump_length: int = 10, jump_n_sample: int = 10, device=None
    ):

        num_inference_steps = min(self.num_train_timesteps, num_inference_steps)
        self.num_inference_steps = num_inference_steps

        timesteps = []

        jumps = {}
        for j in range(0, num_inference_steps - jump_length, jump_length):
            jumps[j] = jump_n_sample - 1

        # 20 times until 80 + 9 times forward and backward = 200
        t = num_inference_steps
        while t >= 1:
            t = t - 1
            timesteps.append(t)

            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    timesteps.append(t)

        timesteps = np.array(timesteps) * (self.num_train_timesteps // self.num_inference_steps)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(self, model_output, timestep, sample, original_image, mask):
        t = timestep
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # Compute alphas & betas for timestep t
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        # Sample random noise
        device = model_output.device
        noise = torch.randn(model_output.shape).to(device)

        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = torch.clamp(
            (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5, -1, 1
        )

        # Compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output

        # Compute x_{t-1} of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_unknown_part = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

        # Algorithm 1 Line 5 https://arxiv.org/pdf/2201.09865.pdf
        prev_known_part = (alpha_prod_t_prev**0.5) * original_image + (
            (1 - alpha_prod_t_prev) ** 0.5
        ) * noise

        # Algorithm 1 Line 8 https://arxiv.org/pdf/2201.09865.pdf
        pred_prev_sample = mask * prev_known_part + (1.0 - mask) * prev_unknown_part

        return pred_prev_sample

    def undo_step(self, sample, timestep):
        n = self.num_train_timesteps // self.num_inference_steps

        for i in range(n):
            beta = self.betas[timestep + i]
            noise = torch.randn(sample.shape).to(sample.device)

            # 10. Algorithm 1 Line 10 https://arxiv.org/pdf/2201.09865.pdf
            sample = (1 - beta) ** 0.5 * sample + beta**0.5 * noise

        return sample
