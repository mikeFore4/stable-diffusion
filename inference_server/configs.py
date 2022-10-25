import pydantic
import os

class ModelConfig(pydantic.BaseModel):
    yaml_config: str = "configs/stable-diffusion/v1-inference.yaml"
    checkpoint: str = "weights/sd-v1-4.ckpt"
    plms: bool = False
    batch_size: int = 1

    @pydantic.validator('yaml_config')
    def config_validator(cls, yaml_config: str):
        if not os.path.exists(yaml_config):
            raise FileNotFoundError("Try using full path for model config")

        return yaml_config

    @pydantic.validator('checkpoint')
    def checkpoint_validator(cls, checkpoint: str):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError("Try using full path for model checkpoint")

        return checkpoint

class TextPromptConfig(pydantic.BaseModel):
    prompt: str
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    height: int = 512
    width: int = 512
    latent_channels: int = 4
    downsampling_factor: int = 8
    n_samples: int = 1
    scale: float = 7.5
    precision: str = "autocast"
    output_dir: str = None

    @pydantic.validator('precision')
    def validator(cls, precision):
        if precision not in ["full", "autocast"]:
            raise ValueError("precision field must be either 'full' or 'autocast'")
        return precision

class FullConfig(TextPromptConfig, ModelConfig):
    pass

