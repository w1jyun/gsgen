from .stable_diffusion_prompt import StableDiffusionPromptProcessor
from .controlnet_prompt import ControlNetPromptProcessor
from .deep_floyd_prompt import DeepFloydPromptProcessor

prompt_processors = dict(
    stable_diffusion=StableDiffusionPromptProcessor,
    controlnet=ControlNetPromptProcessor,
    deep_floyd=DeepFloydPromptProcessor,
)


def get_prompt_processor(cfg, **kwargs):
    try:
        return prompt_processors[cfg.type](cfg, **kwargs)
    except KeyError:
        raise NotImplementedError(f"Prompt processor {cfg.type} not implemented.")
