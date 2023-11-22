from . import BaseGuidance
import torch
import torch.nn.functional as F

from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from torchvision.utils import save_image

from .dreamtime import Timestep

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from rich.console import Console
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

console = Console()

# from prompt.prompt_processors import BasePromptProcessor, PromptEmbedding


# class StableDiffusionPromptProcessor(BasePromptProcessor):
#     def prepare_text_encoder(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.pretrained_model_name_or_path, subfolder="tokenizer"
#         )
#         self.text_encoder = CLIPTextModel.from_pretrained(
#             self.pretrained_model_name_or_path,
#             subfolder="text_encoder",
#             device_map="auto",
#         )

#     def encode_prompts(self, prompts):
#         with torch.no_grad():
#             print(prompts)
#             tokens = self.tokenizer(
#                 prompts,
#                 padding="max_length",
#                 max_length=self.tokenizer.model_max_length,
#                 return_tensors="pt",
#             ).to(self.device)
#             # print(tokens.input_ids.device)
#             text_embeddings = self.text_encoder(tokens.input_ids)[0]

#         return text_embeddings


class ControlNetGuidance(BaseGuidance):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.weights_dtype = (
            torch.float32
        )

        if self.cfg.keep_complete_pipeline:
            pipe_kwargs = {
                "torch_dtype": self.weights_dtype,
            }
        else:
            pipe_kwargs = {
                "torch_dtype": self.weights_dtype,
                "cache_dir": "./.cache",
            }

        if self.cfg.repeat_until_success:
            success = False
            while not success:
                try:
                    self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=self.weights_dtype)
                    self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        controlnet=self.controlnet,
                        **pipe_kwargs,
                    ).to(self.device)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    console.print(".", end="")
                else:
                    success = True
                    break
        else:
            self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=self.weights_dtype)
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=self.controlnet,
                **pipe_kwargs,
            ).to('cuda')

        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        # self.unet = nn.DataParallel(unet).to(self.device)

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        # TODO: make this configurable
        scheduler = self.cfg.scheduler.type.lower()
        if scheduler == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )
        elif scheduler == "pndm":
            self.scheduler = PNDMScheduler(**self.cfg.scheduler.args)
        else:
            raise NotImplementedError(f"Scheduler {scheduler} not implemented")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.step = 0
        self.max_steps = self.cfg.max_steps
        self.set_min_max_steps()
        self.grad_clip_val = None
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.Timestep = Timestep(num_of_timestep=self.max_t_step, num_of_iters=self.max_steps)
        if self.cfg.enable_attention_slicing:
            # enable GPU VRAM saving, reference: https://huggingface.co/stabilityai/stable-diffusion-2
            self.pipe.enable_attention_slicing(1)

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self):
        min_step_percent = C(self.cfg.min_step_percent, self.step, self.max_steps)
        max_step_percent = C(self.cfg.max_step_percent, self.step, self.max_steps)
        self.min_t_step = int(self.num_train_timesteps * min_step_percent)
        self.max_t_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents,
        t,
        encoder_hidden_states,
        controlnet_cond,
    ):
        input_dtype = latents.dtype
        latents = latents.to(self.weights_dtype)
        t = t.to(self.weights_dtype)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            guess_mode=True,
            return_dict=False,
        )
        return self.unet(
            latents,
            t,
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs):
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype).cuda()).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents,
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    def compute_grad_sds(
        self,
        latents,
        t,
        prompt_embedding,
        elevation,
        azimuth,
        camera_distances,
        controlnet_cond,
    ):
        batch_size = elevation.shape[0]
        controlnet_cond = self.pipe.prepare_image(controlnet_cond, 512, 512, batch_size, 1, 'cuda', self.weights_dtype, do_classifier_free_guidance=True).to(dtype=self.weights_dtype)
        if prompt_embedding.use_perp_negative:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_embedding.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.use_view_dependent_prompt
            )
            with torch.no_grad():
                noise = torch.randn_like(latents).cuda()
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_cond,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
            noise_pred = noise_pred.to(self.weights_dtype)
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_embedding.get_text_embedding(
                elevation, azimuth, camera_distances, self.cfg.use_view_dependent_prompt
            )

            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents).cuda()  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_cond,
                ).cuda()

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1).cuda()
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1).cuda()
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_embedding.use_perp_negative,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        # pred_rgb_ = self.decode_latents(latents)
        # noisy = self.decode_latents(latents_noisy)
        # pred_minus_noise = self.decode_latents(grad)
        # target = self.decode_latents(latents - grad)
        # arr = [pred_rgb_, noisy, controlnet_cond, pred_minus_noise, target]
        # viz_images = torch.cat(arr)
        # print('output_')
        # save_image(viz_images, 'output_.png')

        return grad, guidance_eval_utils, controlnet_cond

    def forward(
        self,
        rgb,
        prompt_embedding,
        elevation,
        azimuth,
        camera_distance,
        controlnet_cond,
        rgb_as_latents=False,
        guidance_eval=False,
        mask=None,
        bg=None,
        idx=0,
        **kwargs,
    ):
        bs = rgb.shape[0]
        rgb_BCHW_512 = None

        bg_color = (bg[0][0][0])
        # rgb_np = rgb.detach().cpu().numpy()[0]

        # # print(rgb_np.shape)
        # rgb_np_mask = np.zeros_like(rgb_np)
        # H = rgb_np.shape[0]
        # W = rgb_np.shape[1]
        # for h in range(H):
        #     for w in range(W):
        #         if not np.array_equal(rgb_np[h][w], bg_color):
        #             rgb_np_mask[h][w] = [1.0,1.0,1.0]
        # rgb_np_mask = np.where(np.array_equal(rgb_np, bg_color), np.array([0,0,0]), np.array([1,1,1]))
        sketch_mask = (mask[0].astype(np.float32))
        # loss_mask = 0.5 * F.mse_loss(torch.from_numpy(rgb_np_mask, requires_grad = True), torch.from_numpy(sketch_mask, requires_grad = True), reduction="sum") / (bs * 3)
        # if idx % 10 == 0:
        #     plt.imsave('sketch_mask%d.png'%idx, sketch_mask)
        #     plt.close()
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        ts = []
        for b in range(bs):
            t_ = self.Timestep.timestep(idx * bs + b)
            ts.append(t_)
        t = torch.tensor(ts, dtype=torch.long, device='cuda')

        # t = torch.randint(
        #     self.min_t_step,
        #     self.max_t_step + 1,
        #     [bs],
        #     dtype=torch.long,
        #     device=self.device,
        # )

        grad, guidance_eval_utils, controlnet_cond = self.compute_grad_sds(
            latents, t, prompt_embedding, elevation, azimuth, camera_distance, controlnet_cond
        )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (latents - grad).detach()
        target_img = self.decode_latents(target)
        target_img_masked = target_img.clone().detach()[0]
        target_img_masked = target_img_masked.permute(1, 2, 0)
        H = sketch_mask.shape[0]
        W = sketch_mask.shape[1]
        for h in range(H):
            for w in range(W):
                if sum(sketch_mask[h][w]) == 0:
                    target_img_masked[h][w] = bg_color

        target_img_masked = target_img_masked.permute(2, 0, 1)
        target_img_masked = target_img_masked.unsqueeze(dim=0)
        target = self.encode_images(target_img_masked)
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / bs
        loss_sds_each = 0.5 * F.mse_loss(latents, target, reduction="none").sum(
            dim=[1, 2, 3]
        )

        guidance_out = {
            "loss_sds": loss_sds,
            # "loss_mask": loss_mask,
            "loss_sds_each": loss_sds_each,
            "grad_norm": grad.norm(),
            "min_t_step": self.min_t_step,
            "max_t_step": self.max_t_step,
        }

        if idx % 10 == 0:
          target_img = self.decode_latents(target)
          rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
          )
          arr = [rgb_BCHW_512, target_img]
          viz_images = torch.cat(arr)
          save_image(viz_images, 'output_%d.png'%idx)

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distance
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    # def step(self, epoch: int, step: int):
    #     if self.cfg.grad_clip is not None:
    #         self.grad_clip_val = C(self.cfg.grad_clip, epoch, step)

    # vanilla scheduler use constant min max steps
    # self.set_min_max_steps()

    def update(self, step):
        self.step = step
        self.set_min_max_steps()
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, step, self.max_steps)

    def log(self, writer, step):
        writer.add_scalar("guidance/min_step", self.min_t_step, step)
        writer.add_scalar("guidance/max_step", self.max_t_step, step)
