from transformers import AutoImageProcessor, ConvNextModel

from . import register_vision_tower
from .base import VisionTower

@register_vision_tower('convnext')      
class DINOv2VisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._vision_tower = ConvNextModel(cfg)
        self._image_processor = AutoImageProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    
    def forward(self, x, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[-1]
        b,c,h,w = image_features.shape
        image_features = image_features.view(b,c,-1).transpose(1,2)
        image_features = image_features
        return image_features
  
