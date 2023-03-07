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
import time
import unittest

import numpy as np
import paddle

from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import load_numpy, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


class StableDiffusion2VPredictionPipelineFastTests(unittest.TestCase):

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @property
    def dummy_cond_unet(self):
        paddle.seed(0)
        model = UNet2DConditionModel(block_out_channels=(32, 64),
            layers_per_block=2, sample_size=32, in_channels=4, out_channels
            =4, down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
            up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
            cross_attention_dim=32, attention_head_dim=(2, 4),
            use_linear_projection=True)
        return model

    @property
    def dummy_vae(self):
        paddle.seed(0)
        model = AutoencoderKL(block_out_channels=[32, 64], in_channels=3,
            out_channels=3, down_block_types=['DownEncoderBlock2D',
            'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D',
            'UpDecoderBlock2D'], latent_channels=4, sample_size=128)
        return model

    @property
    def dummy_text_encoder(self):
        paddle.seed(0)
        config = CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size
            =32, intermediate_size=37, layer_norm_eps=1e-05,
            num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
            vocab_size=1000, hidden_act='gelu', projection_dim=64)
        return CLIPTextModel(config).eval()

    def test_stable_diffusion_v_pred_ddim(self):
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
            beta_schedule='scaled_linear', clip_sample=False,
            set_alpha_to_one=False, prediction_type='v_prediction')
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            'hf-internal-testing/tiny-random-clip')
        sd_pipe = StableDiffusionPipeline(unet=unet, scheduler=scheduler,
            vae=vae, text_encoder=bert, tokenizer=tokenizer, safety_checker
            =None, feature_extractor=None, requires_safety_checker=False)
        sd_pipe = sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = 'A painting of a squirrel eating a burger'
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=6.0,
            num_inference_steps=2, output_type='np')
        image = output.images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe([prompt], generator=generator,
            guidance_scale=6.0, num_inference_steps=2, output_type='np',
            return_dict=False)[0]
        image_slice = image[(0), -3:, -3:, (-1)]
        image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.6424, 0.6109, 0.494, 0.5088, 0.4984, 
            0.4525, 0.5059, 0.5068, 0.4474])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
            ) < 0.01

    def test_stable_diffusion_v_pred_k_euler(self):
        unet = self.dummy_cond_unet
        scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=
            0.012, beta_schedule='scaled_linear', prediction_type=
            'v_prediction')
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            'hf-internal-testing/tiny-random-clip')
        sd_pipe = StableDiffusionPipeline(unet=unet, scheduler=scheduler,
            vae=vae, text_encoder=bert, tokenizer=tokenizer, safety_checker
            =None, feature_extractor=None, requires_safety_checker=False)
        sd_pipe = sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = 'A painting of a squirrel eating a burger'
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=6.0,
            num_inference_steps=2, output_type='np')
        image = output.images
        generator = paddle.Generator().manual_seed(0)
        image_from_tuple = sd_pipe([prompt], generator=generator,
            guidance_scale=6.0, num_inference_steps=2, output_type='np',
            return_dict=False)[0]
        image_slice = image[(0), -3:, -3:, (-1)]
        image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4616, 0.5184, 0.4887, 0.5111, 0.4839, 
            0.48, 0.5119, 0.5263, 0.4776])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
            ) < 0.01

    @unittest.skipIf(torch_device != 'cuda', 'This test requires a GPU')
    def test_stable_diffusion_v_pred_fp16(self):
        """Test that stable diffusion v-prediction works with fp16"""
        unet = self.dummy_cond_unet
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
            beta_schedule='scaled_linear', clip_sample=False,
            set_alpha_to_one=False, prediction_type='v_prediction')
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained(
            'hf-internal-testing/tiny-random-clip')
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        unet = unet.to(dtype=paddle.float16)
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        vae = vae.to(dtype=paddle.float16)
        """Class Method: *.to, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>        bert = bert.to(dtype=paddle.float16)
        sd_pipe = StableDiffusionPipeline(unet=unet, scheduler=scheduler,
            vae=vae, text_encoder=bert, tokenizer=tokenizer, safety_checker
            =None, feature_extractor=None, requires_safety_checker=False)
        sd_pipe = sd_pipe
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = 'A painting of a squirrel eating a burger'
        generator = paddle.Generator().manual_seed(0)
        image = sd_pipe([prompt], generator=generator, num_inference_steps=
            2, output_type='np').images
        assert image.shape == (1, 64, 64, 3)


@slow
@require_paddle_gpu
class StableDiffusion2VPredictionPipelineIntegrationTests(unittest.TestCase):

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_v_pred_default(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2')
        sd_pipe = sd_pipe
        sd_pipe.enable_attention_slicing()
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = 'A painting of a squirrel eating a burger'
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=7.5,
            num_inference_steps=20, output_type='np')
        image = output.images
        image_slice = image[(0), 253:256, 253:256, (-1)]
        assert image.shape == (1, 768, 768, 3)
        expected_slice = np.array([0.1868, 0.1922, 0.1527, 0.1921, 0.1908, 
            0.1624, 0.1779, 0.1652, 0.1734])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_v_pred_upcast_attention(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1', paddle_dtype=paddle.float16)
        sd_pipe = sd_pipe
        sd_pipe.enable_attention_slicing()
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = 'A painting of a squirrel eating a burger'
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, guidance_scale=7.5,
            num_inference_steps=20, output_type='np')
        image = output.images
        image_slice = image[(0), 253:256, 253:256, (-1)]
        assert image.shape == (1, 768, 768, 3)
        expected_slice = np.array([0.4209, 0.4087, 0.4097, 0.4209, 0.386, 
            0.4329, 0.428, 0.4324, 0.4187])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05

    def test_stable_diffusion_v_pred_euler(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            'stabilityai/stable-diffusion-2', subfolder='scheduler')
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2', scheduler=scheduler)
        sd_pipe = sd_pipe
        sd_pipe.enable_attention_slicing()
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = 'A painting of a squirrel eating a burger'
        generator = paddle.Generator().manual_seed(0)
        output = sd_pipe([prompt], generator=generator, num_inference_steps
            =5, output_type='numpy')
        image = output.images
        image_slice = image[(0), 253:256, 253:256, (-1)]
        assert image.shape == (1, 768, 768, 3)
        expected_slice = np.array([0.1781, 0.1695, 0.1661, 0.1705, 0.1588, 
            0.1699, 0.2005, 0.1589, 0.1677])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_v_pred_dpm(self):
        """
        TODO: update this test after making DPM compatible with V-prediction!
        """
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            'stabilityai/stable-diffusion-2', subfolder='scheduler')
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2', scheduler=scheduler)
        sd_pipe = sd_pipe
        sd_pipe.enable_attention_slicing()
        sd_pipe.set_progress_bar_config(disable=None)
        prompt = 'a photograph of an astronaut riding a horse'
        generator = paddle.Generator().manual_seed(0)
        image = sd_pipe([prompt], generator=generator, guidance_scale=7.5,
            num_inference_steps=5, output_type='numpy').images
        image_slice = image[(0), 253:256, 253:256, (-1)]
        assert image.shape == (1, 768, 768, 3)
        expected_slice = np.array([0.3303, 0.3184, 0.3291, 0.33, 0.3256, 
            0.3113, 0.2965, 0.3134, 0.3192])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_attention_slicing_v_pred(self):
>>>        torch.cuda.reset_peak_memory_stats()
        model_id = 'stabilityai/stable-diffusion-2'
        pipe = StableDiffusionPipeline.from_pretrained(model_id,
            paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        prompt = 'a photograph of an astronaut riding a horse'
        pipe.enable_attention_slicing()
        generator = paddle.Generator().manual_seed(0)
        output_chunked = pipe([prompt], generator=generator, guidance_scale
            =7.5, num_inference_steps=10, output_type='numpy')
        image_chunked = output_chunked.images
        mem_bytes = paddle.device.cuda.max_memory_allocated()>>>        torch.cuda.reset_peak_memory_stats()
        assert mem_bytes < 5.5 * 10 ** 9
        pipe.disable_attention_slicing()
        generator = paddle.Generator().manual_seed(0)
        output = pipe([prompt], generator=generator, guidance_scale=7.5,
            num_inference_steps=10, output_type='numpy')
        image = output.images
        mem_bytes = paddle.device.cuda.max_memory_allocated()        assert mem_bytes > 5.5 * 10 ** 9
        assert np.abs(image_chunked.flatten() - image.flatten()).max() < 0.001

    def test_stable_diffusion_text2img_pipeline_v_pred_default(self):
        expected_image = load_numpy(
            'https://huggingface.co/datasets/hf-internal-testing/ppdiffusers-images/resolve/main/sd2-text2img/astronaut_riding_a_horse_v_pred.npy'
            )
        pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2')
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=None)
        prompt = 'astronaut riding a horse'
        generator = paddle.Generator().manual_seed(0)
        output = pipe(prompt=prompt, guidance_scale=7.5, generator=
            generator, output_type='np')
        image = output.images[0]
        assert image.shape == (768, 768, 3)
        assert np.abs(expected_image - image).max() < 0.075

    def test_stable_diffusion_text2img_pipeline_v_pred_fp16(self):
        expected_image = load_numpy(
            'https://huggingface.co/datasets/hf-internal-testing/ppdiffusers-images/resolve/main/sd2-text2img/astronaut_riding_a_horse_v_pred_fp16.npy'
            )
        pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2', paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        prompt = 'astronaut riding a horse'
        generator = paddle.Generator().manual_seed(0)
        output = pipe(prompt=prompt, guidance_scale=7.5, generator=
            generator, output_type='np')
        image = output.images[0]
        assert image.shape == (768, 768, 3)
        assert np.abs(expected_image - image).max() < 0.75

    def test_stable_diffusion_text2img_intermediate_state_v_pred(self):
        number_of_steps = 0

>>>        def test_callback_fn(step: int, timestep: int, latents: torch.
            FloatTensor) ->None:
            test_callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 0:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 96, 96)
                latents_slice = latents[(0), -3:, -3:, (-1)]
                expected_slice = np.array([0.7749, 0.0325, 0.5088, 0.1619, 
                    0.3372, 0.3667, -0.5186, 0.686, 1.4326])
                assert np.abs(latents_slice.flatten() - expected_slice).max(
                    ) < 0.05
            elif step == 19:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 96, 96)
                latents_slice = latents[(0), -3:, -3:, (-1)]
                expected_slice = np.array([1.3887, 1.0273, 1.7266, 0.0726, 
                    0.6611, 0.1598, -1.0547, 0.1522, 0.0227])
                assert np.abs(latents_slice.flatten() - expected_slice).max(
                    ) < 0.05
        test_callback_fn.has_been_called = False
        pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2', paddle_dtype=paddle.float16)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        prompt = 'Andromeda galaxy in a bottle'
        generator = paddle.Generator().manual_seed(0)
        pipe(prompt=prompt, num_inference_steps=20, guidance_scale=7.5,
            generator=generator, callback=test_callback_fn, callback_steps=1)
        assert test_callback_fn.has_been_called
        assert number_of_steps == 20

    def test_stable_diffusion_low_cpu_mem_usage_v_pred(self):
        pipeline_id = 'stabilityai/stable-diffusion-2'
        start_time = time.time()
        pipeline_low_cpu_mem_usage = StableDiffusionPipeline.from_pretrained(
            pipeline_id, paddle_dtype=paddle.float16)
        pipeline_low_cpu_mem_usage
        low_cpu_mem_usage_time = time.time() - start_time
        start_time = time.time()
        _ = StableDiffusionPipeline.from_pretrained(pipeline_id,
            paddle_dtype='float16', low_cpu_mem_usage=False)
        normal_load_time = time.time() - start_time
        assert 2 * low_cpu_mem_usage_time < normal_load_time

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading_v_pred(
        self):
        paddle.device.cuda.empty_cache()

        pipeline_id = 'stabilityai/stable-diffusion-2'
        prompt = 'Andromeda galaxy in a bottle'
        pipeline = StableDiffusionPipeline.from_pretrained(pipeline_id,
            paddle_dtype=paddle.float16)
        pipeline = pipeline
        pipeline.enable_attention_slicing(1)
        pipeline.enable_sequential_cpu_offload()
        generator = paddle.Generator().manual_seed(0)
        _ = pipeline(prompt, generator=generator, num_inference_steps=5)
        mem_bytes = paddle.device.cuda.max_memory_allocated()        assert mem_bytes < 2.8 * 10 ** 9