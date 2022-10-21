from configs import ModelConfig, TextPromptConfig, FullConfig
from omegaconf import OmegaConf
import torch

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import argparse

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def instantiate_model(opt: ModelConfig):
    if opt.laion400m:
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.checkpoint = "models/ldm/text2img-large/model.ckpt"

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(opt.config, opt.checkpiont)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    return model, sampler

def make_prediction(opt: TextPromptConfig, model, sampler):
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                data = [opt.prompt]
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning([""])
                c = model.get_learned_conditioning(prompts)
                shape = [opt.latent_channels, opt.height // opt.downsampling_factor, opt.width // opt.downsampling_factor]

                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x_T=None)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                x_sample = 255. * rearrange(x_checked_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))

    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--n_iter', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--latent_channels', type=int)
    parser.add_argument('--downsampling_factor', type=int)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--precision', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--plms', action='store_true')
    parser.add_argument('--laion400m', action='store_true')
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    cfg = FullConfig(**vars(args))

    model, sampler = instantiate_model(cfg)
    img = make_prediction(cfg, model, sampler)
    img.save(args.output_path)

if __name__=='__main__':
    main()
