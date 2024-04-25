from typing import Any, Dict
from typing import Optional, OrderedDict

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
    
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image    

__all__ = [
    "ad_openai_clip_vitl_patch14_336"
]

def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }
    
default_cfgs = {
    "ad_openai_clip_vitl_patch14_336" : _cfg(
        hf_hub_id = "chitradrishti/adlike",
        input_size=(3, 336, 336)
    )
}

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
    
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        _convert_image_to_rgb,
        ToTensor(),
        Resize((n_px,n_px), antialias=True),   
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
class PostClipProjectorBlock(nn.Module):
    def __init__(
        self,
        dim,
        out_dim : Optional[int] = None,
        dropout = 0.3,
        *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        if out_dim is None:
            out_dim = dim // 2

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def create_post_clip_projector(embed_dim, num_proj_layers, num_classes, dropout=0.):
    dims = [embed_dim//(2**k) for k in range(0, num_proj_layers)]
    out_dims = dims[1:] + [num_classes]  
    proj = torch.nn.Sequential(*[PostClipProjectorBlock(dim, out_dim) for dim, out_dim in zip(dims, out_dims)])
    return proj
    
    
def load_ad_openai_clip(cfg_key, vision_model, projector, device):

    cfg = default_cfgs[cfg_key]

    checkpoint_path = hf_hub_download(repo_id=cfg["hf_hub_id"], filename=cfg_key,)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    clip_vision_state_dict = OrderedDict()
    projector_state_dict = OrderedDict()

    clip_tok = "clip_model."
    clip_tok_s = len(clip_tok)
    
    proj_tok = "projector."
    proj_tok_s = len(proj_tok)
        
    for (k, v) in state_dict["state_dict"].items():
        k : str = k
        if k.startswith(clip_tok):
            clip_vision_state_dict[k[clip_tok_s:]] = v
        elif k.startswith(proj_tok):
            projector_state_dict[k[proj_tok_s:]] = v
            
                            
    vision_model.load_state_dict(state_dict = clip_vision_state_dict)
    projector.load_state_dict(state_dict=projector_state_dict)

    return vision_model, projector


class AdProbability(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return 1 - torch.sigmoid(x)    

    
def ad_openai_clip_vitl_patch14_336(eval_probab = True, device = "cpu", **kwargs):
    """ Image Advertisement Model based on OpenAI Clip Model ViT-L/14@336px.
    """
    
    from clip.model import VisionTransformer
    
    cfg_key = "ad_openai_clip_vitl_patch14_336"
    
    model_args = dict(input_resolution=336, patch_size=14, width=1024, layers=24, heads=16, output_dim=768)
    projector_args = dict(embed_dim=768, num_proj_layers=4, num_classes=1)
    
    vision_model = VisionTransformer(**model_args)
    projector = create_post_clip_projector(**projector_args)

    load_ad_openai_clip(cfg_key, vision_model, projector, device)
    
    d = OrderedDict()
    d["visual"] = vision_model
    d["proj"] = projector
    
    if eval_probab:
        d["probab"] = AdProbability()

    model = torch.nn.Sequential(d)
    preprocess = _transform(vision_model.input_resolution)
    
    return model.eval(), preprocess