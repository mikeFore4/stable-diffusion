from torch import autocast
from diffusers import StableDiffusionPipeline
import argparse
import os

def generate_image(prompt,
            output_dir,
            n_samples,
            version,
            steps,
            height,
            width):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pipe = StableDiffusionPipeline.from_pretrained(
            f"CompVis/stable-diffusion-v1-{version}",
            use_auth_token=True
            )

    prompt = [prompt] * n_samples
    with autocast('cuda'):
        images = pipe(prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width)['sample']

    for i,img in enumerate(images):
        img.save(os.path.join(output_dir,f'{i}.png'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('-o', '--output-dir', type=str)
    parser.add_argument('-v','--version',default=4,type=int)
    parser.add_argument('--steps',default=50, type=int)
    parser.add_argument('-n','--n_samples',default=1,type=int)
    parser.add_argument('--height',default=512,type=int)
    parser.add_argument('--width',default=512,type=int)

    args = parser.parse_args()

    generate_image(**vars(args))
