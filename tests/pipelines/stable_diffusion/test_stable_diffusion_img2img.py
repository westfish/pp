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
from ppdiffusers_test.test_pipelines_common import PipelineTesterMixin

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, nightly, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusionImg2ImgPipelineFastTests(PipelineTesterMixin, unittest
    .TestCase):
    pipeline_class = StableDiffusionImg2ImgPipeline

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(block_out_channels=(32, 64),
            layers_per_block=2, sample_size=32, in_channels=4, out_channels
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
        generator = paddle.Generator().manual_seed(seed)

        inputs = {'prompt': 'A painting of a squirrel eating a burger',
            'image': image, 'generator': generator, 'num_inference_steps': 
            2, 'guidance_scale': 6.0, 'output_type': 'numpy'}
        return inputs

    def test_stable_diffusion_img2img_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4492, 0.3865, 0.4222, 0.5854, 0.5139, 
            0.4379, 0.4193, 0.48, 0.4218])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_img2img_negative_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        negative_prompt = 'french fries'
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4065, 0.3783, 0.405, 0.5266, 0.4781, 
            0.4252, 0.4203, 0.4692, 0.4365])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_img2img_multiple_init_images(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs['prompt'] = [inputs['prompt']] * 2
        inputs['image'] = inputs['image'].tile(repeat_times=[2, 1, 1, 1])
        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]
        assert image.shape == (2, 32, 32, 3)
        expected_slice = np.array([0.5144, 0.4447, 0.4735, 0.6676, 0.5526, 
            0.5454, 0.645, 0.5149, 0.4689])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_img2img_k_lms(self):
        components = self.get_dummy_components()
        components['scheduler'] = LMSDiscreteScheduler(beta_start=0.00085,
            beta_end=0.012, beta_schedule='scaled_linear')
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4367, 0.4986, 0.4372, 0.6706, 0.5665, 
            0.444, 0.5864, 0.6019, 0.5203])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_stable_diffusion_img2img_num_images_per_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionImg2ImgPipeline(**components)
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
class StableDiffusionImg2ImgPipelineSlowTests(unittest.TestCase):

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype='float32',
        seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(
            'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/sketch-mountains-input.png'
            )
        inputs = {'prompt':
            'a fantasy landscape, concept art, high resolution', 'image':
            init_image, 'generator': generator, 'num_inference_steps': 3,
            'strength': 0.75, 'guidance_scale': 7.5, 'output_type': 'numpy'}
        return inputs

    def test_stable_diffusion_img2img_default(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4', safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.43, 0.4662, 0.493, 0.399, 0.4307, 
            0.4525, 0.3719, 0.4064, 0.3923])
        assert np.abs(expected_slice - image_slice).max() < 0.001

    def test_stable_diffusion_img2img_k_lms(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4', safety_checker=None)
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config
            )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.0389, 0.0346, 0.0415, 0.029, 0.0218, 
            0.021, 0.0408, 0.0567, 0.0271])
        assert np.abs(expected_slice - image_slice).max() < 0.001

    def test_stable_diffusion_img2img_ddim(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4', safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.0593, 0.0607, 0.0851, 0.0582, 0.0636, 
            0.0721, 0.0751, 0.0981, 0.0781])
        assert np.abs(expected_slice - image_slice).max() < 0.001

    def test_stable_diffusion_img2img_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: paddle.Tensor
            ) ->None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.4958, 0.5107, 1.1045, 2.7539,
                    4.668, 3.832, 1.5049, 1.8633, 2.6523])
                assert np.abs(latents_slice.flatten() - expected_slice).max(
                    ) < 0.05
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 96)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([-0.4956, 0.5078, 1.0918, 2.752, 
                    4.6484, 3.8125, 1.5146, 1.8633, 2.6367])
                assert np.abs(latents_slice.flatten() - expected_slice).max(
                    ) < 0.05
        callback_fn.has_been_called = False
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4', safety_checker=None,
            paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(dtype='float16')
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 2


    def test_stable_diffusion_pipeline_with_model_offloading(self):
        paddle.device.cuda.empty_cache()

        inputs = self.get_inputs(dtype='float16')
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4', safety_checker=None,
            paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        pipe(**inputs)
        mem_bytes = paddle.device.cuda.max_memory_allocated()        
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4', safety_checker=None,
            paddle_dtype=paddle.float16)
        paddle.device.cuda.empty_cache()

        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        _ = pipe(**inputs)
        mem_bytes_offloaded = paddle.device.cuda.max_memory_allocated()
        assert mem_bytes_offloaded < mem_bytes
        for module in (pipe.text_encoder, pipe.unet, pipe.vae):
            assert module.place == 'cpu'

    def test_stable_diffusion_img2img_pipeline_multiple_of_8(self):
        init_image = load_image(
            'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg'
            )
        init_image = init_image.resize((760, 504))
        model_id = 'CompVis/stable-diffusion-v1-4'
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,
            safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        prompt = 'A fantasy landscape, trending on artstation'
        generator = paddle.Generator().manual_seed(0)
        output = pipe(prompt=prompt, image=init_image, strength=0.75,
            guidance_scale=7.5, generator=generator, output_type='np')
        image = output.images[0]
        image_slice = image[255:258, 383:386, -1]
        assert image.shape == (504, 760, 3)
        expected_slice = np.array([0.9393, 0.95, 0.9399, 0.9438, 0.9458, 
            0.94, 0.9455, 0.9414, 0.9423])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005


@nightly
@require_paddle_gpu
class StableDiffusionImg2ImgPipelineNightlyTests(unittest.TestCase):

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype='float32',
        seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image(
            'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/sketch-mountains-input.png'
            )
        inputs = {'prompt':
            'a fantasy landscape, concept art, high resolution', 'image':
            init_image, 'generator': generator, 'num_inference_steps': 50,
            'strength': 0.75, 'guidance_scale': 7.5, 'output_type': 'numpy'}
        return inputs

    def test_img2img_pndm(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5')
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/stable_diffusion_1_5_pndm.npy'
            )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_img2img_ddim(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5')
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/stable_diffusion_1_5_ddim.npy'
            )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_img2img_lms(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5')
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.
            scheduler.config)
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/stable_diffusion_1_5_lms.npy'
            )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_img2img_dpm(self):
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5')
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe
            .scheduler.config)
        sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        inputs['num_inference_steps'] = 30
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/stable_diffusion_1_5_dpm.npy'
            )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001
