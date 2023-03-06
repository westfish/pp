# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import paddle

from ppdiffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from ppdiffusers.utils.testing_utils import require_paddle_gpu, slow


class DDPMPipelineFastTests(unittest.TestCase):
    @property
    def dummy_uncond_unet(self):
        paddle.seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    def test_fast_inference(self):
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler()
        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="numpy").images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = ddpm(generator=generator, num_inference_steps=2, output_type="numpy", return_dict=False)[0]
        image_slice = image[(0), -3:, -3:, (-1)]
        image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.9956, 0.5785, 0.4675, 0.993, 0.0, 1.0, 0.001199, 0.0002648, 0.0005101])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01

    def test_inference_predict_sample(self):
        unet = self.dummy_uncond_unet
        scheduler = DDPMScheduler(prediction_type="sample")
        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.set_progress_bar_config(disable=None)
        if torch_device == "mps":
            _ = ddpm(num_inference_steps=1)
        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=2, output_type="numpy").images
        generator = paddle.Generator().manual_seed(0)
        image_eps = ddpm(generator=generator, num_inference_steps=2, output_type="numpy")[0]
        image_slice = image[(0), -3:, -3:, (-1)]
        image_eps_slice = image_eps[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 32, 32, 3)
        tolerance = 0.01 if torch_device != "mps" else 0.03
        assert np.abs(image_slice.flatten() - image_eps_slice.flatten()).max() < tolerance


@slow
@require_paddle_gpu
class DDPMPipelineIntegrationTests(unittest.TestCase):
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"
        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDPMScheduler.from_pretrained(model_id)
        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        image = ddpm(generator=generator, output_type="numpy").images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.42, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
