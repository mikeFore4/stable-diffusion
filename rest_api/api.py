from fastapi import FastAPI
from inference_server.configs import TextPromptConfig
import json

#def gen_img(prompt, ddim_steps, ddim_eta, height, width, latent_channels,
#        downsampling_factor, n_samples, scale, precision, output_dir):

app = FastAPI()

@app.post('/generate')
def gen_img(req: TextPromptConfig):
    with open('../inputs/new_input.json','w') as f:
        json.dump(json.loads(req.json()), f)

    return "Successful"

