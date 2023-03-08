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
import random
import unittest

import numpy as np
import paddle
from PIL import Image
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusionInstructPix2PixPipelineFastTests(PipelineTesterMixin,
    unittest.TestCase):
    pipeline_class = StableDiffusionInstructPix2PixPipeline

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(block_out_channels=(32, 64),
            layers_per_block=2, sample_size=32, in_channels=8, out_channels
            =4, down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
            up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
            cross_attention_dim=32)
        scheduler = PNDMScheduler(skip_prk_steps=True)
        paddle.seed(0)
        vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3,
            out_channels=3, down_block_types=['DownEncoderBlock2D',
            'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
            'UpDecoderBlock2D'], latent_channels=4)
        paddle.seed(0)
        text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
            hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
            num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
            vocab_size=1000)
        text_encoder = CLIPTextModel(text_encoder_config).eval()
        tokenizer = CLIPTokenizer.from_pretrained(
            'hf-internal-testing/tiny-random-clip')
        components = {'unet': unet, 'scheduler': scheduler, 'vae': vae,
            'text_encoder': text_encoder, 'tokenizer': tokenizer,
            'safety_checker': None, 'feature_extractor': None}
        return components

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        generator = paddle.Generator().manual_seed(seed)

        inputs = {'prompt': 'A painting of a squirrel eating a burger',
            'image': image, 'generator': generator, 'num_inference_steps': 
            2, 'guidance_scale': 6.0, 'image_guidance_scale': 1,
            'output_type': 'numpy'}
        return inputs

    def test_stable_diffusion_pix2pix_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.7318, 0.3723, 0.4662, 0.623, 0.577, 
            0.5014, 0.4281, 0.555, 0.4813])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_pix2pix_negative_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        negative_prompt = 'french fries'
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.7323, 0.3688, 0.4611, 0.6255, 0.5746, 
            0.5017, 0.433, 0.5553, 0.4827])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_pix2pix_multiple_init_images(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs['prompt'] = [inputs['prompt']] * 2
        image = np.array(inputs['image']).astype(np.float32) / 255.0
        image = paddle.to_tensor(data=image).unsqueeze(axis=0)
        image = image.transpose(perm=[0, 3, 1, 2])
        inputs['image'] = image.tile(repeat_times=[2, 1, 1, 1])
        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]
        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array([0.606, 0.5712, 0.5099, 0.598, 0.5805, 
            0.7205, 0.6793, 0.554, 0.5607])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_pix2pix_euler(self):
        components = self.get_dummy_components()
        components['scheduler'] = EulerAncestralDiscreteScheduler(beta_start
            =0.00085, beta_end=0.012, beta_schedule='scaled_linear')
        sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        slice = [round(x, 4) for x in image_slice.flatten().tolist()]
        print(','.join([str(x) for x in slice]))
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.726, 0.3902, 0.4868, 0.585, 0.5672, 
            0.511, 0.3906, 0.551, 0.4846])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_pix2pix_num_images_per_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs).images
        assert images.shape == (1, 32, 32, 3)
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs['prompt'] = [inputs['prompt']] * batch_size
        images = sd_pipe(**inputs).images
        assert images.shape == (batch_size, 32, 32, 3)
        num_images_per_prompt = 2
        inputs = self.get_dummy_inputs()
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt
            ).images
        assert images.shape == (num_images_per_prompt, 32, 32, 3)
        batch_size = 2
        inputs = self.get_dummy_inputs()
        inputs['prompt'] = [inputs['prompt']] * batch_size
        images = sd_pipe(**inputs, num_images_per_prompt=num_images_per_prompt
            ).images
        assert images.shape == (batch_size * num_images_per_prompt, 32, 32, 3)


@slow
@require_paddle_gpu
class StableDiffusionInstructPix2PixPipelineSlowTests(unittest.TestCase):

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed=seed)
        image = load_image(
            'https://huggingface.co/datasets/ppdiffusers/test-arrays/resolve/main/stable_diffusion_pix2pix/example.jpg'
            )
        inputs = {'prompt': 'turn him into a cyborg', 'image': image,
            'generator': generator, 'num_inference_steps': 3,
            'guidance_scale': 7.5, 'image_guidance_scale': 1.0,
            'output_type': 'numpy'}
        return inputs

    def test_stable_diffusion_pix2pix_default(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            'timbrooks/instruct-pix2pix', safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5902, 0.6015, 0.6027, 0.5983, 0.6092, 
            0.6061, 0.5765, 0.5785, 0.5555])
        assert np.abs(expected_slice - image_slice).max() < 0.001

    def test_stable_diffusion_pix2pix_k_lms(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            'timbrooks/instruct-pix2pix', safety_checker=None)
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config
            )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.6578, 0.6817, 0.6972, 0.6761, 0.6856, 
            0.6916, 0.6428, 0.6516, 0.6301])
        assert np.abs(expected_slice - image_slice).max() < 0.001

    def test_stable_diffusion_pix2pix_ddim(self):
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            'timbrooks/instruct-pix2pix', safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3828, 0.3834, 0.3818, 0.3792, 0.3865, 
            0.3752, 0.3792, 0.3847, 0.3753])
        assert np.abs(expected_slice - image_slice).max() < 0.001

    def test_stable_diffusion_pix2pix_intermediate_state(self):
        number_of_steps = 0
        def callback_fn(step: int, timestep: int, latents: paddle.Tensor
            ) ->None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[(0), -3:, -3:, -1]
                expected_slice = np.array([-0.2463, -0.4644, -0.9756, 
                    1.5176, 1.4414, 0.7866, 0.9897, 0.8521, 0.7983])
                assert np.abs(latents_slice.flatten() - expected_slice).max(
                    ) < 0.05
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[(0), -3:, -3:, -1]
                expected_slice = np.array([-0.2644, -0.4626, -0.9653, 
                    1.5176, 1.4551, 0.7686, 0.9805, 0.8452, 0.8115])
                assert np.abs(latents_slice.flatten() - expected_slice).max(
                    ) < 0.05
        callback_fn.has_been_called = False
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            'timbrooks/instruct-pix2pix', safety_checker=None, paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 3



    def test_stable_diffusion_pix2pix_pipeline_multiple_of_8(self):
        inputs = self.get_inputs()
        inputs['image'] = inputs['image'].resize((504, 504))
        model_id = 'timbrooks/instruct-pix2pix'
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
            safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        output = pipe(**inputs)
        image = output.images[0]
        image_slice = image[255:258, 383:386, -1]
        assert image.shape == (504, 504, 3)
        expected_slice = np.array([0.2726, 0.2529, 0.2664, 0.2655, 0.2641, 
            0.2642, 0.2591, 0.2649, 0.259])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005
