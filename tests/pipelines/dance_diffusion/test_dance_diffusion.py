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

import gc
import unittest

import numpy as np
import paddle
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from ppdiffusers import DanceDiffusionPipeline, IPNDMScheduler, UNet1DModel
from ppdiffusers.utils import slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class DanceDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DanceDiffusionPipeline
    test_attention_slicing = False
    test_cpu_offload = False

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet1DModel(
            block_out_channels=(32, 32, 64),
            extra_in_channels=16,
            sample_size=512,
            sample_rate=16000,
            in_channels=2,
            out_channels=2,
            flip_sin_to_cos=True,
            use_timestep_embedding=False,
            time_embedding_type="fourier",
            mid_block_type="UNetMidBlock1D",
            down_block_types=("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
        )
        scheduler = IPNDMScheduler()
        components = {"unet": unet, "scheduler": scheduler}
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = {"batch_size": 1, "generator": generator, "num_inference_steps": 4}
        return inputs

    def test_dance_diffusion(self):
        components = self.get_dummy_components()
        pipe = DanceDiffusionPipeline(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)
        audio = output.audios
        audio_slice = audio[0, -3:, -3:]
        assert audio.shape == (1, 2, components["unet"].sample_size)
        expected_slice = np.array([-0.7265, 1.0, -0.8388, 0.1175, 0.9498, -1.0])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 0.01


@slow
@require_paddle_gpu
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_dance_diffusion(self):
        pipe = DanceDiffusionPipeline.from_pretrained("harmonai/maestro-150k")
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=100, audio_length_in_s=4.096)
        audio = output.audios
        audio_slice = audio[0, -3:, -3:]
        assert audio.shape == (1, 2, pipe.unet.sample_size)
        expected_slice = np.array([-0.0192, -0.0231, -0.0318, -0.0059, 0.0002, -0.002])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 0.01

    def test_dance_diffusion_fp16(self):
        pipe = DanceDiffusionPipeline.from_pretrained("harmonai/maestro-150k", paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=100, audio_length_in_s=4.096)
        audio = output.audios
        audio_slice = audio[0, -3:, -3:]
        assert audio.shape == (1, 2, pipe.unet.sample_size)
        expected_slice = np.array([-0.0367, -0.0488, -0.0771, -0.0525, -0.0444, -0.0341])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 0.01
