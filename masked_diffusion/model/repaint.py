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

        original_image = image.to(device=self.device)
        mask_image = mask_image.to(device=self.device)

        image = torch.randn(image.shape, device=self.device)

        self.scheduler.set_timesteps(
            self.num_inference_steps, self.jump_length, self.jump_n_sample, self.device
        )

        self.unet.expand_dims = True

        logger.info("Inpainting...")
        t_last = self.scheduler.timesteps[0] + 1
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
            if t < t_last:
                # predict the noise residual
                model_output = self.unet(image, t.unsqueeze(0))
                self.unet.expand_dims = False

                # compute previous image: x_t -> x_t-1
                image = self.scheduler.step(
                    model_output, t.squeeze(), image, original_image, mask_image
                ).prev_sample

            else:
                # compute the reverse: x_t-1 -> x_t
                image = self.scheduler.undo_step(image, t_last.squeeze())
            t_last = t

        image = torch.mean(image, dim=1, keepdim=True)

        return image
