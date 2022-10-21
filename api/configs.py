from pydantic import BaseModel

class ModelConfig(BaseModel):
    config: str = "configs/stable-diffusion/v1-inference.yaml"
    checkpoint: str = "models/ldm/stable-diffusion-v1/model.ckpt"
    plms: bool = False
    laion400m: bool = False

class TextPromptConfig(BaseModel):
    prompt: str
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    n_iter: int = 2
    height: int = 512
    width: int = 512
    latent_channels: int = 4
    downsampling_factor: int = 8
    n_samples: int = 1
    scale: float = 7.5
    precision: str = "autocast"

    @pydantic.validator("precision")
    @classmethod
    def precision_validator(self, value):
        if value not in ["full", "autocast"]:
            raise ValueError("precision field must be either 'full' or 'autocast'")
        return value

class FullConfig(TextPromptConfig, ModelConfig):
    pass

