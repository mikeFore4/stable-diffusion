from pydantic import BaseModel
from fastapi import FastAPI

from utils import instantiate_model, make_prediction
from configs import ModelConfig, TextPromptConfig, FullConfig

app = FastAPI()
