from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPTokenizer, CLIPTextModel
from dataset import EEGImageNetDataset
from PIL import Image
import argparse
import torch
import os
from tqdm import tqdm

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
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    dic = {}
    blip_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(blip_id, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(blip_id, use_safetensors=True, local_files_only=True).to(
        device)
    for image_name in tqdm(dataset.images):
        image_path = os.path.join("../data/imageNet_images", image_name.split("_")[0], image_name)
        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(images=raw_image, return_tensors="pt").to(device)
        generation_config = {
            "max_length": 200,  # 增加描述的最大长度
            "num_beams": 20,  # Beam Search的广度
            "temperature": 0.5,  # 生成随机性的控制
            "top_k": 0,  # top-k采样
            "top_p": 0.9,  # top-p采样
            "repetition_penalty": 2.0,  # 重复惩罚
            "do_sample": True  # 启用采样
        }
        out = model.generate(**inputs, **generation_config)
        caption = processor.decode(out[0], skip_special_tokens=True)
        with open(os.path.join(args.output_dir, "caption.txt"), "a") as f:
            f.write(f"{image_name}\t{caption}\n")
        tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer",
                                                  local_files_only=True)
        text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
                                                     use_safetensors=True, local_files_only=True).to(device)
        inputs = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(inputs.input_ids.to(device))[0]
        dic[image_name] = text_embeddings
    torch.save(dic, os.path.join(args.output_dir, "clip_embeddings.pth"))
