import modules.scripts as scripts
import gradio as gr
import torch
import clip
import yaml
import os
import tempfile
from datetime import datetime
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
        gr.Markdown("### 🖼️ Image Caption")

        with gr.Row():
            with gr.Column(variant="panel"):
                image = gr.Image(
                    type="pil",
                    label="Upload / Drag & Drop Image",
                    source="upload",
                    height=420,
                    show_download_button=True,
                )

                choice = gr.Dropdown(
                    choices=["Cocomix", "NTP"],
                    value="Cocomix",
                    label="Model",
                )

                with gr.Row():
                    btn = gr.Button("Generate caption", variant="primary")
                    clear_btn = gr.ClearButton([image], value="Clear image")

            with gr.Column(variant="panel"):
                gr.Markdown("#### Caption")
                caption_display = gr.Textbox(
                    value="",
                    show_label=False,
                    interactive=True,  # let user tweak before saving
                )

                with gr.Row():
                    copy_btn = gr.Button("Copy to clipboard")
                    with gr.Column(scale=1):
                        download_file = gr.File(label="Download caption (.txt)", visible=False)

                gr.Markdown("#### History")
                history_gallery = gr.Gallery(
                    label="Generated (image + caption)",
                    show_label=True,
                    height=300,
                    columns=4,
                    preview=True,
                    show_download_button=False,
                )
                # keep the gallery entries in a State
                history_state = gr.State([])

        # --- Wiring ---
        # 1) Main generation
        gen_event = btn.click(
            fn=get_caption,
            inputs=[choice, image],
            outputs=caption_display,
            show_progress="full",
        )

        # 2) Then save caption to a .txt and show as downloadable file
        gen_event.then(
            fn=save_caption_to_txt,
            inputs=caption_display,
            outputs=download_file,
        )

        # 3) Then append to gallery history (and persist in state)
        gen_event.then(
            fn=update_history,
            inputs=[image, caption_display, history_state],
            outputs=[history_gallery, history_state],
        )

        # 4) Copy-to-clipboard (pure JS; no Python call). Works in modern browsers.
        #    This reads the textbox value and writes it to the clipboard.
        copy_btn.click(
            fn=None,
            inputs=caption_display,
            outputs=None,
            _js="(text) => navigator.clipboard?.writeText?.(text)"
        )

        return [(ui_component, "Image Caption", "image_caption_tab")]

def save_caption_to_txt(caption: str):
    """
    Create a temporary file named caption-<timestamp>.txt
    and return its path so the browser shows a Save As… dialog.
    """
    if not caption:
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"caption-{timestamp}.txt"

    # Create a temp file with the desired filename
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(caption)

    return gr.update(value=path, visible=True)

def update_history(image, caption, gallery_state):
    """
    Append (image, caption) to the history gallery.
    - gallery_state is a list of entries already shown in the Gallery.
    - Gallery accepts either images or (image, caption) tuples.
    """
    if image is None or not caption:
        return gallery_state, gallery_state  # no change
    # Ensure the state is a list
    items = list(gallery_state or [])
    items.insert(0, (image, caption))
    return items, items

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
