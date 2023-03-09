export LD_LIBRARY_PATH=/root/anaconda3/envs/benchmark/lib:$LD_LIBRARY_PATH
export RUN_SLOW=True
export RUN_NIGHTLY=True
export FROM_HF_HUB=True
export FROM_DIFFUSERS=True
export TO_DIFFUSERS=True
export HF_HOME="/root/mttest/test_caches"
export PPNLP_HOME="/root/mttest/test_caches"

# pytest -v tests/test_config.py
# pytest -v tests/test_hub_utils.py
# pytest -v tests/test_outputs.py
# pytest -v tests/test_scheduler.py
# pytest -v tests/test_training.py
# pytest -v tests/test_utils.py
# pytest -v tests/test_layers_utils.py
# CUDA_VISIBLE_DEVICES=3 /root/anaconda3/envs/benchmark/bin/python -m pytest -v tests/pipelines/stable_diffusion_2/test*
# CUDA_VISIBLE_DEVICES=3 /root/anaconda3/envs/benchmark/bin/python -m pytest -v tests/pipelines/altdiffusion/test*

# CUDA_VISIBLE_DEVICES=3 /root/anaconda3/envs/benchmark/bin/python -m pytest -v tests/pipelines/altdiffusion/test_alt_diffusion.py::AltDiffusionPipelineFastTests::test_save_load_local
# CUDA_VISIBLE_DEVICES=3 /root/anaconda3/envs/benchmark/bin/python -m pytest -v tests/pipelines/altdiffusion/test_alt_diffusion.py


# CUDA_VISIBLE_DEVICES=3 /root/anaconda3/envs/benchmark/bin/python -m pytest -v tests/pipelines/dance_diffusion/test_dance_diffusion.py


# CUDA_VISIBLE_DEVICES=3 /root/anaconda3/envs/benchmark/bin/python -m pytest -v tests/pipelines/paint_by_example


# CUDA_VISIBLE_DEVICES=2 /root/anaconda3/envs/benchmark/bin/python -m pytest -v tests/pipelines/semantic_stable_diffusion
# export CUDA_LAUNCH_BLOCKING=1
# CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/benchmark/bin/python -m pytest -v  tests/pipelines/semantic_stable_diffusion/test_semantic_diffusion.py::SemanticDiffusionPipelineIntegrationTests::test_positive_guidance

# CUDA_VISIBLE_DEVICES=2 /root/anaconda3/envs/benchmark/bin/python -m pytest -v tests/pipelines/semantic_stable_diffusion/test_semantic_diffusion.py::SafeDiffusionPipelineFastTests::test_semantic_diffusion_no_safety_checker

CUDA_VISIBLE_DEVICES=2 /root/anaconda3/envs/benchmark/bin/python -m pytest -v tests/pipelines/altdiffusion/test_alt_diffusion_img2img.py