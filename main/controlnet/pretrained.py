import json

import torch
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from stable_audio_tools.models.utils import load_ckpt_state_dict

from main.controlnet.factory import create_model_from_config

from huggingface_hub import hf_hub_download

def get_pretrained_controlnet_model(name: str, cavp_config_path:str, cavp_ckpt_path:str, clap_ckpt_path:str, depth_factor=0.5, flag_modality="audio", use_cavp=True, sample_size=88200, sample_rate=44100):
    model_config_path = hf_hub_download(name, filename="model_config.json", repo_type='model')

    with open(model_config_path) as f:
        model_config = json.load(f)
    model_config["model_type"] = "diffusion_cond_controlnet"
    model_config["sample_size"] = sample_size
    model_config["sample_rate"] = sample_rate
    model_config["model"]["diffusion"]['config']["controlnet_depth_factor"] = depth_factor
    model_config["model"]["diffusion"]["type"] = "dit_controlnet"

    cavp_config = {
        "id": "frames",
        "type": "cavp_frames",
        "config": {
            "config_path": cavp_config_path,
            "ckpt_path": cavp_ckpt_path
        }
    }

    if flag_modality == "audio":
        clap_audio_config = {
                        "id": "audio",
                        "type": "clap_audio",
                        "config": {
                            "audio_model_type": "HTSAT-tiny",
                            "enable_fusion": True,
                            "clap_ckpt_path": clap_ckpt_path
                        }
                    }
    else:
        clap_text_config = {
                        "id": "text",
                        "type": "clap_text",
                        "config": {
                            "audio_model_type": "HTSAT-tiny",
                            "enable_fusion": True,
                            "clap_ckpt_path": clap_ckpt_path
                        }
                    }
    
    if use_cavp:
        model_config["model"]["conditioning"]["configs"].append(cavp_config)
    if flag_modality == "audio":
        model_config["model"]["conditioning"]["configs"].append(clap_audio_config)
    else:
        model_config["model"]["conditioning"]["configs"].append(clap_text_config)

    controlnet_conditioner_config = {"id": "envelope",
                                     "type": "pretransform",
                                     "config": {"sample_rate": model_config["sample_rate"],
                                                "output_dim": model_config["model"]["pretransform"]["config"]["latent_dim"],
                                               "pretransform_config": model_config["model"]["pretransform"]}}
    model_config["model"]['conditioning']['configs'].append(controlnet_conditioner_config)

    ids_to_remove = ["prompt"]
    model_config["model"]["conditioning"]["configs"] = [config for config in model_config["model"]["conditioning"]["configs"] if config["id"] not in ids_to_remove]

    list_cond_cond_ids = []

    if flag_modality == "audio":
        list_cond_cond_ids.append("audio")
    else:
        list_cond_cond_ids.append("text")

    if use_cavp:
        list_cond_cond_ids.append("frames")
        
    model_config["model"]["diffusion"]["cross_attention_cond_ids"] = list_cond_cond_ids

    model_config["model"]["diffusion"]["global_cond_ids"] = ["seconds_start", "seconds_total"]
    model_config["model"]["diffusion"]['controlnet_cond_ids'] = ["envelope"]
    model = create_model_from_config(model_config)

    # Try to download the model.safetensors file first, if it doesn't exist, download the model.ckpt file
    try:
        model_ckpt_path = hf_hub_download(name, filename="model.safetensors", repo_type='model')
    except Exception as e:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type='model')

    state_dict = load_ckpt_state_dict(model_ckpt_path)

    model.load_state_dict(state_dict, strict=False)
    state_dict_controlnet = {k.split('model.model.')[-1]: v for k, v in state_dict.items() if k.startswith('model.model')}
    model.model.controlnet.load_state_dict(state_dict_controlnet, strict=False)

    state_dict_pretransform = {k: v for k, v in state_dict.items() if k.startswith('pretransform.')}
    model.conditioner.conditioners['envelope'].load_state_dict(state_dict_pretransform)

    return model, model_config


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from main.data.dataset_musdb import create_musdb_dataset, collate_fn_conditional

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_config = get_pretrained_controlnet_model("stabilityai/stable-audio-open-1.0")
    model = model.cuda()

    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]

    dataset = create_musdb_dataset("../../data/musdb18hq/train.tar",
                                      sample_rate=44100,
                                      chunk_dur=47.57)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            pin_memory=False,
                            drop_last=True,
                            collate_fn=collate_fn_conditional,
                            num_workers=0)
    data_x, data_y, data_z = next(iter(dataloader))

    conditioning = [{
        "audio": data_y.unsqueeze(1).repeat_interleave(2, dim=1).cuda(),
        "prompt": data_z[0],
        "seconds_start": 0,
        "seconds_total": 40
    }]

    # Generate stereo audio
    output = generate_diffusion_cond(
         model,
         steps=100,
         cfg_scale=7,
         conditioning=conditioning,
         sample_size=sample_size,
         sigma_min=0.3,
         sigma_max=500,
         sampler_type="dpmpp-3m-sde",
         device=device
    )

