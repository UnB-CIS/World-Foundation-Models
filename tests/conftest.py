import pytest
from omegaconf import OmegaConf


@pytest.fixture
def vae_cfg(tmp_path):
    return OmegaConf.create(
        {
            "device": "cpu",
            "experiment_name": "test_vae",
            "compute": {"num_workers": 0, "pin_memory": False},
            "data": {
                "videos_folder": "data/scenario_1/videos",
                "frames_root": str(tmp_path / "frames"),
                "frame_step": 1,
                "img_size": 16,
                "batch_size": 4,
                "test_split": 0.2,
            },
            "training": {
                "num_epochs": 2,
                "lr": 1e-3,
                "max_norm": 2.0,
                "warmup_epochs": 1,
                "scheduler_patience": 1,
                "scheduler_factor": 0.5,
            },
            "paths": {
                "save_dir": str(tmp_path / "checkpoints" / "vae"),
                "samples_dir": str(tmp_path / "samples" / "vae"),
            },
        }
    )


@pytest.fixture
def fusion_cfg(tmp_path):
    return OmegaConf.create(
        {
            "device": "cpu",
            "experiment_name": "test_fusion",
            "compute": {"num_workers": 0, "pin_memory": False},
            "data": {
                "source_dir": str(tmp_path / "processed"),
                "test_split": 0.2,
                "batch_size": 4,
            },
            "training": {
                "num_epochs": 2,
                "lr": 1e-3,
                "max_norm": 1.0,
                "scheduler_patience": 1,
                "scheduler_factor": 0.5,
            },
            "paths": {
                "save_dir": str(tmp_path / "checkpoints" / "fusion"),
                "samples_dir": str(tmp_path / "samples" / "fusion"),
            },
        }
    )


@pytest.fixture
def action_encoder_cfg(tmp_path):
    return OmegaConf.create(
        {
            "device": "cpu",
            "experiment_name": "test_action_encoder",
            "compute": {"num_workers": 0, "pin_memory": False},
            "scenario": {
                "screen_width": 800,
                "screen_height": 600,
                "type_encoding": {"mouse_down": [1.0], "none": [0.0]},
                "object_encoding": {"ball": [1.0], "none": [0.0]},
            },
            "data": {
                "num_samples": 40,
                "positive_ratio": 0.5,
                "test_split": 0.2,
                "batch_size": 8,
            },
            "model": {
                "input_dim": 4,
                "latent_dim": 8,
                "output_dim": 4,
            },
            "training": {
                "num_epochs": 2,
                "lr": 1e-3,
                "max_norm": 1.0,
            },
            "paths": {
                "save_dir": str(tmp_path / "checkpoints" / "action_encoder"),
                "samples_dir": str(tmp_path / "samples" / "action_encoder"),
            },
        }
    )
