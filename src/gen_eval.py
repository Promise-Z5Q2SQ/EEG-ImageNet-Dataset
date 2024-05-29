import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import numpy as np
from PIL import Image
from tqdm import tqdm
from de_feat_cal import de_feat_cal
from dataset import EEGImageNetDataset
from model.mlp_sd import MLPMapper
from utilities import *

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer",
                                          local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
                                             use_safetensors=True, local_files_only=True,
                                             torch_dtype=torch.bfloat16).to(device)
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True,
                                    local_files_only=True, torch_dtype=torch.bfloat16).to(device)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True,
                                            local_files_only=True, torch_dtype=torch.bfloat16).to(device)
scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler",
                                          local_files_only=True)
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 50  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.Generator(device=device).manual_seed(42)


def diffusion(embeddings):
    batch_size = embeddings.size()[0]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=tokenizer.model_max_length,
                             return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, embeddings])
    latents = torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8), generator=generator,
                          device=device, dtype=torch.bfloat16)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        generated_images = vae.decode(latents).sample
    generated_images = ((generated_images / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
    return generated_images


def model_init(args):
    if args.model.lower() == 'mlp_sd':
        _model = MLPMapper()
    else:
        raise ValueError(f"Couldn't find the model {args.model}")
    return _model


clip_embeddings = torch.load(os.path.join("../output/", "clip_embeddings.pth"))


def save_generated_images(args, dataloader, model):
    model.to(device)
    model.eval()
    with (torch.no_grad()):
        for index, (inputs, labels) in enumerate(dataloader):
            labels = torch.stack([clip_embeddings[image_name] for image_name in labels]).squeeze()
            labels = labels.to(device=device, dtype=torch.bfloat16)
            inputs = inputs.to(device=device)
            embeddings = model(inputs).to(dtype=torch.bfloat16)
            generated_images = diffusion(embeddings)
            for i, image in enumerate(generated_images):
                file = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
                file.save(os.path.join(args.output_dir, f"generated_s{args.subject}/",
                                       f"{i + 1 + index * args.batch_size}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", required=True, help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset(args)
    eeg_data = np.stack([i[0].numpy() for i in dataset], axis=0)
    # extract frequency domain features
    de_feat = de_feat_cal(eeg_data, args)
    dataset.add_frequency_feat(de_feat)

    model = model_init(args)
    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, str(args.pretrained_model))))
    if args.model.lower() == 'mlp_sd':
        dataset.use_frequency_feat = True
        dataset.use_image_label = True
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        save_generated_images(args, dataloader, model)
