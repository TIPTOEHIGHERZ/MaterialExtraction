import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
import copy


class FeatureMatcher(nn.Module):
    def __init__(self, vision_backbone: CLIPVisionTransformer, device='cpu'):
        super().__init__()
        # maybe frozen this or using low learning rate to avoid too much domain change?
        self.vision_backbone: CLIPVisionTransformer = vision_backbone
        self.vision_backbone_mask: CLIPVisionTransformer = copy.deepcopy(vision_backbone)

        self.device = device
        self.to(device)
        return
    
    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        if image.shape[1] != mask.shape[1]:
            mask = mask[:, 0, ...].unsqueeze(1)
        
        image_masked = image * (1 - mask)
        features: BaseModelOutputWithPooling = self.vision_backbone(image)
        features_masked: BaseModelOutputWithPooling = self.vision_backbone_mask(image_masked)

        return features.pooler_output, features_masked.pooler_output
    
    @staticmethod
    def from_pretrained(fp: str, device='cpu'):
        clip_model = CLIPModel.from_pretrained(fp)
        return FeatureMatcher(clip_model.vision_model, device=device)

    def from_pretrained_personalized(fp: str):
        pass
    
    def convert_features(self, image: torch.Tensor):
        out = self.vision_backbone_mask(image)
        return out
    
    def frozen(self):
        for param in self.vision_backbone.parameters():
            param.requires_grad = False

        return


class FeatureProjection(nn.Module):
    def __init__(self, matcher: FeatureMatcher, projection_dim: int):
        super().__init__()
        self.matcher = matcher
        # will be trained alone
        self.linear = nn.Linear(matcher.vision_backbone.config.hidden_size, projection_dim)
        return
    
    def forward(self, image: torch.Tensor):
        features = self.matcher(image)
        out = self.linear(features)
        return out
    
    def frozen(self):
        for param in self.matcher.parameters():
            param.requires_grad = False

        return
