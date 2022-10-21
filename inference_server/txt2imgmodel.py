from omegaconf import OmegaConf
from einops import rearrange
from PIL import Image
from configs import TextPromptConfig
import torch
import os
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import numpy as np

class Txt2ImgModel():

    def __init__(self, config: str = "configs/stable-diffusion/v1-inference.yaml", checkpoint: str = "weights/sd-v1-4.ckpt", plms: bool = False, laion400m:
            bool = False):
        self.config = self._load_config(config)
        self.checkpoint = self._load_checkpoint(checkpoint)
        self.plms = plms

        self.model = self._load_model()
        self.sampler = self._load_sampler()

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError("Try using full path for model config")

        return OmegaConf.load(f"{opt.config}")

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Try using full path for model checkpoint")

        return checkpoint_path

    def _load_model(self, verbose=False):
        pl_sd = torch.load(self.checkpoint, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(self.config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        return model

    def _load_sampler(self):
        if self.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        return sampler

    def generate_image(self, opt: TextPromptConfig):
        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        data = [opt.prompt]
        batch_size = 1
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    uc = None
                    if opt.scale != 1.0:
                        uc = self.model.get_learned_conditioning(batch_size * [""])
                    c = self.model.get_learned_conditioning(prompts)
                    shape = [opt.latent_channels, opt.height // opt.downsampling_factor, opt.width // opt.downsampling_factor]
                    samples_ddim, _ = self.sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=batch_size,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=None)
                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    x_sample = 255. * rearrange(x_checked_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))

        return img

