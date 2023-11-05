import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from .prompt_processors import BasePromptProcessor, PromptEmbedding


class ControlNetPromptProcessor(BasePromptProcessor):
    def prepare_text_encoder(self, guidance_model=None):
        self.pipe = guidance_model.pipe

    def encode_prompts(self, prompts):
        return self.pipe._encode_prompt(prompts, device='cpu', num_images_per_prompt=1, do_classifier_free_guidance=True)
    
    def update(self, step):
        pass
