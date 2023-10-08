from __future__ import annotations

from functools import cached_property

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    ControlNetModel
)
import torch

from asdff.base import AdPipelineBase


class AdPipeline(AdPipelineBase, StableDiffusionPipeline):
    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionPipeline


class AdCnPipeline(AdPipelineBase, StableDiffusionControlNetPipeline):
    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionControlNetInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionControlNetPipeline




class AdCnPreloadPipe(AdPipelineBase):

    def __init__(self, pipe = None):
      if pipe != None:
          self.inpaint_pipeline(pipe)

    def txt2img_class(self, pipe_txt):
      self.txt2img_class = pipe_txt
      return self.txt2img_class

    def inpaint_pipeline(self, pipe):

      controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")
      self.preload = True
      self.inpaint_pipeline = StableDiffusionControlNetInpaintPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=pipe.unet,
            controlnet=controlnet,
            scheduler=pipe.scheduler,
            safety_checker=pipe.safety_checker,
            feature_extractor=pipe.feature_extractor,
            requires_safety_checker=pipe.config.requires_safety_checker,
        )

      return self.inpaint_pipeline

    @property
    def txt2img_class(self):
        return StableDiffusionControlNetPipeline
