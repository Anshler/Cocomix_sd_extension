import modules.scripts as scripts
import gradio as gr
import torch
import clip
import yaml
from modules import script_callbacks
from transformers import  GPT2Tokenizer
from typing import List, Optional, Union, Tuple, Dict, Any
from huggingface_hub import hf_hub_download

from scripts.clipcap_utils import merge_dicts, dict_to_obj, ParserObject
from scripts.clipcap_model import ClipCaptionModel, generate_clip_prefix

previous_choice = 'Cocomix'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_model, preprocess, model, tokenizer = None, None, None, None
args, cfg = None, None
prefix_dim = 768


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            with gr.Column(variant='panel'):
                image = gr.Image(type='pil', label="Image", source='upload')
            with gr.Column(variant='panel'):
                    
                choice = gr.Radio(choices = ["Cocomix", "NTP"], value = 'Cocomix', label="ClipCap model")
                
                btn = gr.Button("Generate caption", variant='primary').style(
                    full_width=False
                    )
                
                caption_display = gr.Textbox(
                    default="",
                    label="Caption",
                    interactive = False
                    )

        btn.click(
            get_caption,
            inputs = [choice, image],
            outputs = caption_display,
        )

        return [(ui_component, "Image Caption", "image_caption_tab")]

def get_caption(choice, image = None):
    global previous_choice
    global device, clip_model, preprocess, model, tokenizer
    global prefix_dim
    global args, cfg
    if image is not None:
        torch.cuda.empty_cache()
        
        # things that need to be re-declare if changed, they are heavy
        if choice != previous_choice:
            previous_choice = choice
            change_model = True
        else:
            change_model = False
            
        if change_model or model is None:
            change_model = False
            repo_id = 'Anshler/clip-cocomix'
            pre_cfg_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
            
            # Select model
            if choice == 'Cocomix':
              add_cfg_path = hf_hub_download(repo_id=repo_id, filename="gpt2_69m_cocomix.yaml")
              clipcap_model_path = hf_hub_download(repo_id=repo_id, filename="cocomix.pt")
              base_gpt2_path = hf_hub_download(repo_id=repo_id, filename="cocomix.safetensors")
            else:
              add_cfg_path = hf_hub_download(repo_id=repo_id, filename="gpt2_69m_ntp.yaml")
              clipcap_model_path = hf_hub_download(repo_id=repo_id, filename="ntp.pt")
              base_gpt2_path = hf_hub_download(repo_id=repo_id, filename="ntp.safetensors")
            
            # Create config object from yaml
            with open(pre_cfg_path, "r") as file:
              pre_cfg = yaml.safe_load(file)
            with open(add_cfg_path, "r") as file:
              add_cfg = yaml.safe_load(file)

            cfg = dict_to_obj(merge_dicts(pre_cfg, add_cfg))

            # Create config object defined within code
            args = ParserObject()
            gpt2_config = hf_hub_download(repo_id=repo_id, filename="config.json")
            args.gpt2_type = {'base_gpt2_path': base_gpt2_path, 'gpt2_config': gpt2_config}

            # Create model
            clip_model, preprocess = clip.load(name='ViT-L/14', device=device)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            model = ClipCaptionModel(prefix_length=args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                            num_layers=args.num_layers, gpt2_type=args.gpt2_type,
                            prefix_only=args.prefix_only, prefix_concept_enable=args.prefix_concept_enable, cfg=cfg)

            model.load_state_dict(torch.load(clipcap_model_path, map_location=torch.device('cpu')), strict=False)
            model = model.eval()
            model = model.to('cuda')
      
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image)
            prefix_embed = model.clip_project(prefix.to(dtype=torch.float32)).reshape(1, args.prefix_length, -1)
            
        caption = generate_clip_prefix(model, tokenizer, embed=prefix_embed).split('.')[0].strip()+' .'
        
        return caption
        
    else:
        return ''

script_callbacks.on_ui_tabs(on_ui_tabs)
