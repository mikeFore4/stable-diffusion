from configs import ModelConfig, TextPromptConfig
import json
from datetime import datetime
from time import sleep
from txt2imgmodel import Txt2ImgModel
import os
import argparse

class InferenceEngine():

    def __init__(self, input_folder: str = 'inputs', output_folder: str =
            'outputs', sleep_time: int = 2, model_config_path: str = None):
        self.sleep_time = sleep_time
        self.input_folder = input_folder
        self.output_folder = output_folder

        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        if model_config_path is None:
            model_config_path = self._check_for_model_config()

        self.model = self._load_model(model_config_path)
        print('Model loaded. Inference server ready...')

    def run_inference(self):
        while True:
            inp = self._check_for_input()
            if inp is None:
                sleep(self.sleep_time)
            else:
                txtprompt = self._load_input(inp)
                self._del_input(inp)
                self.model.generate_image(txtprompt)

    def _check_for_model_config(self):
        path = os.path.join(self.input_folder, 'model_config.json')
        if os.path.exists(path):
            return path
        else:
            return None

    def _check_for_input(self):
        for inp_path in os.listdir(self.input_folder):
            if inp_path.endswith('.json'):
                return inp_path

        return None

    def _del_input(self, inp_path):
        os.remove(os.path.join(self.input_folder, inp_path))

        return

    def _load_model(self, config_path=None):
        if config_path is None:
            mod_cfg = {}
        else:
            with open(config_path, 'r') as f:
                mod_cfg = json.load(f)

        mod_cfg = ModelConfig(**mod_cfg)

        return Txt2ImgModel(mod_cfg)

    def _load_input(self, fn):
        with open(os.path.join(self.input_folder, fn), 'r') as f:
            inp = json.load(f)

        inp = TextPromptConfig(**inp)
        inp.output_dir = self.output_folder

        return inp

def main():
    inf_eng = InferenceEngine()
    inf_eng.run_inference()

if __name__=='__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--model_config', type=str)

    #args = parser.parse_args()

    #inf_eng = InferenceEngine(args.model_config)
    main()
