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

import contextlib
import gc
import inspect
import io
import re
import tempfile
import unittest
from typing import Callable, Union

import numpy as np
import paddle

import ppdiffusers
from ppdiffusers import (
    CycleDiffusionPipeline,
    DanceDiffusionPipeline,
    DiffusionPipeline,
    RePaintPipeline,
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
)
from ppdiffusers.utils import logging
from ppdiffusers.utils.import_utils import (
    is_cutlass_fused_multi_head_attention_available,
)
from ppdiffusers.utils.testing_utils import require_paddle


@require_paddle
class PipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline, e.g. saving and loading the pipeline,
    equivalence of dict and tuple outputs, etc.
    """

    allowed_required_args = [
        "source_prompt",
        "prompt",
        "image",
        "mask_image",
        "example_image",
        "class_labels",
        "token_indices",
    ]
    required_optional_params = ["generator", "num_inference_steps", "return_dict"]
    num_inference_steps_args = ["num_inference_steps"]
    test_attention_slicing = True
    test_cpu_offload = False
    test_xformers_attention = True

    def get_generator(self, seed):
        generator = paddle.Generator().manual_seed(seed)
        return generator

    @property
    def pipeline_class(self) -> Union[Callable, DiffusionPipeline]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. See existing pipeline tests for reference."
        )

    def get_dummy_components(self):
        raise NotImplementedError(
            "You need to implement `get_dummy_components(self)` in the child test class. See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self, seed=0):
        raise NotImplementedError(
            "You need to implement `get_dummy_inputs(self, device, seed)` in the child test class. See existing pipeline tests for reference."
        )

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_save_load_local(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            breakpoint()
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, from_diffusers=False)
            pipe_loaded.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 0.0001)

    def test_pipeline_call_implements_required_args(self):
        assert hasattr(self.pipeline_class, "__call__"), f"{self.pipeline_class} should have a `__call__` method"
        parameters = inspect.signature(self.pipeline_class.__call__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        required_parameters.pop("self")
        required_parameters = set(required_parameters)
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        for param in required_parameters:
            if param == "kwargs":
                continue
            assert param in self.allowed_required_args
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        for param in self.required_optional_params:
            assert param in optional_parameters

    def test_inference_batch_consistent(self):
        self._test_inference_batch_consistent()

    def _test_inference_batch_consistent(self, batch_sizes=[2, 4, 13]):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=ppdiffusers.logging.FATAL)
        for batch_size in batch_sizes:
            batched_inputs = {}
            for name, value in inputs.items():
                if name in self.allowed_required_args:
                    if name == "prompt":
                        len_prompt = len(value)
                        batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                        batched_inputs[name][-1] = 2000 * "very long"
                    else:
                        batched_inputs[name] = batch_size * [value]
                elif name == "batch_size":
                    batched_inputs[name] = batch_size
                else:
                    batched_inputs[name] = value
            for arg in self.num_inference_steps_args:
                batched_inputs[arg] = inputs[arg]
            batched_inputs["output_type"] = None
            if self.pipeline_class.__name__ == "DanceDiffusionPipeline":
                batched_inputs.pop("output_type")
            output = pipe(**batched_inputs)
            assert len(output[0]) == batch_size
            batched_inputs["output_type"] = "np"
            if self.pipeline_class.__name__ == "DanceDiffusionPipeline":
                batched_inputs.pop("output_type")
            output = pipe(**batched_inputs)[0]
            assert output.shape[0] == batch_size
        logger.setLevel(level=ppdiffusers.logging.WARNING)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical()

    def _test_inference_batch_single_identical(
        self, test_max_difference=None, test_mean_pixel_difference=None, relax_max_difference=False
    ):
        if self.pipeline_class.__name__ in [
            "CycleDiffusionPipeline",
            "RePaintPipeline",
            "StableDiffusionPix2PixZeroPipeline",
        ]:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=ppdiffusers.logging.FATAL)
        batched_inputs = {}
        batch_size = 3
        for name, value in inputs.items():
            if name in self.allowed_required_args:
                if name == "prompt":
                    len_prompt = len(value)
                    batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                    batched_inputs[name][-1] = 2000 * "very long"
                else:
                    batched_inputs[name] = batch_size * [value]
            elif name == "batch_size":
                batched_inputs[name] = batch_size
            elif name == "generator":
                batched_inputs[name] = [self.get_generator(i) for i in range(batch_size)]
            else:
                batched_inputs[name] = value
        for arg in self.num_inference_steps_args:
            batched_inputs[arg] = inputs[arg]
        if self.pipeline_class.__name__ != "DanceDiffusionPipeline":
            batched_inputs["output_type"] = "np"
        output_batch = pipe(**batched_inputs)
        assert output_batch[0].shape[0] == batch_size
        inputs["generator"] = self.get_generator(0)
        output = pipe(**inputs)
        logger.setLevel(level=ppdiffusers.logging.WARNING)
        if test_max_difference:
            if relax_max_difference:
                diff = np.abs(output_batch[0][0] - output[0][0])
                diff = diff.flatten()
                diff.sort()
                max_diff = np.median(diff[-5:])
            else:
                max_diff = np.abs(output_batch[0][0] - output[0][0]).max()
            assert max_diff < 0.0001
        if test_mean_pixel_difference:
            assert_mean_pixel_difference(output_batch[0][0], output[0][0])

    def test_dict_tuple_outputs_equivalent(self):

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs())[0]
        output_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
        max_diff = np.abs(output - output_tuple).max()
        self.assertLess(max_diff, 0.0001)

    def test_components_function(self):
        init_components = self.get_dummy_components()
        pipe = self.pipeline_class(**init_components)
        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def test_float16_inference(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        for name, module in components.items():
            if hasattr(module, "to"):
                module.to(dtype=paddle.float16)
            components[name] = module
        pipe_fp16 = self.pipeline_class(**components)
        pipe_fp16.set_progress_bar_config(disable=None)
        output = pipe(**self.get_dummy_inputs())[0]
        output_fp16 = pipe_fp16(**self.get_dummy_inputs())[0]
        max_diff = np.abs(output - output_fp16).max()
        self.assertLess(max_diff, 0.01, "The outputs of the fp16 and fp32 pipelines are too different.")

    def test_save_load_float16(self):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "to"):
                module.to(dtype=paddle.float16)
            components[name] = module
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(
                tmpdir, paddle_dtype=paddle.float16, from_diffusers=False
            )
            pipe_loaded.set_progress_bar_config(disable=None)
        for name, component in pipe_loaded.components.items():
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == paddle.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )
        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 0.003, "The output of the fp16 pipeline changed after saving and loading.")

    def test_save_load_optional_components(self):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, from_diffusers=False)
            pipe_loaded.set_progress_bar_config(disable=None)
        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )
        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 0.0001)

    def test_to_device(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipe.to("cpu")
        model_devices = [str(component.device) for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "Place(cpu)" for device in model_devices))
        output_cpu = pipe(**self.get_dummy_inputs())[0]
        self.assertTrue(np.isnan(output_cpu).sum() == 0)
        pipe.to("gpu")
        model_devices = [str(component.device) for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "Place(gpu:0)" for device in model_devices))
        output_cuda = pipe(**self.get_dummy_inputs())[0]
        self.assertTrue(np.isnan(output_cuda).sum() == 0)

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass()

    def _test_attention_slicing_forward_pass(self, test_max_difference=True):
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output_without_slicing = pipe(**inputs)[0]
        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs()
        output_with_slicing = pipe(**inputs)[0]
        if test_max_difference:
            max_diff = np.abs(output_with_slicing - output_without_slicing).max()
            self.assertLess(max_diff, 0.001, "Attention slicing should not affect the inference results")
        assert_mean_pixel_difference(output_with_slicing[0], output_without_slicing[0])

    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass()

    def _test_xformers_attention_forwardGenerator_pass(self, test_max_difference=True):
        if not self.test_xformers_attention:
            return
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output_without_offload = pipe(**inputs)[0]
        pipe.enable_xformers_memory_efficient_attention()
        inputs = self.get_dummy_inputs()
        output_with_offload = pipe(**inputs)[0]
        if test_max_difference:
            max_diff = np.abs(output_with_offload - output_without_offload).max()
            self.assertLess(max_diff, 0.0001, "XFormers attention should not affect the inference results")
        assert_mean_pixel_difference(output_with_offload[0], output_without_offload[0])

    def test_progress_bar(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        inputs = self.get_dummy_inputs()
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )
        pipe.set_progress_bar_config(disable=True)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")


def assert_mean_pixel_difference(image, expected_image):
    image = np.asarray(DiffusionPipeline.numpy_to_pil(image)[0], dtype=np.float32)
    expected_image = np.asarray(DiffusionPipeline.numpy_to_pil(expected_image)[0], dtype=np.float32)
    avg_diff = np.abs(image - expected_image).mean()
    assert avg_diff < 10, f"Error image deviates {avg_diff} pixels on average"
