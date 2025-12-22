import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class EmbeddingMannager:
    def __init__(self, tokenizer: CLIPTokenizer | str):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.string2token = dict()
    
    def add_tokens(self, string: str | list[str]):
        if isinstance(string, str):
            string = [string]
        input_ids = self.tokenizer(string, truncation=True, max_length=77, return_length=True,
                                    return_overflowing_tokens=False, padding="max_length", return_tensors="pt")['input_ids']
        input_ids = input_ids[:, 1]

        for i in range(len(input_ids)):
            self.string2token[string[i]] = input_ids[i]

        return
    
    def __call__(self, text_embeds: torch.Tensor, text_token: torch.Tensor, image_embeds: torch.Tensor):
        device = text_token.device

        for t in self.string2token.values():
            t = t.to(device)
            if t not in text_token:
                # print(t, text_token)
                continue
            
            idx = torch.where(text_token == t)

            start_emebds = text_embeds[:, :idx[1][0], :]
            end_embeds = text_embeds[:, -1:, :]
            text_embeds = torch.concat([start_emebds, image_embeds, end_embeds], dim=1)
        
        return text_embeds
    

class VisionTextEmbedding(nn.Module):
    def __init__(self, tokenizer: str | CLIPTokenizer, text_encoder: str | CLIPTextModel):
        super().__init__()
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(text_encoder) if isinstance(text_encoder, str) else text_encoder
        self.embedding_manager = EmbeddingMannager(self.tokenizer)

        return
    
    def forward(self, prompt: str | list[str], image_embeds: torch.Tensor):
        device = image_embeds.device

        input_ids = self.tokenizer(prompt, truncation=True, max_length=77, return_length=True,
                                   return_overflowing_tokens=False, padding="max_length", return_tensors="pt")['input_ids'].to(device)
        
        text_embeds = self.text_encoder(input_ids)[0]
        text_embeds = self.embedding_manager(text_embeds, input_ids, image_embeds)

        return text_embeds
    
    def add_tokens(self, *args, **kwargs):
        self.embedding_manager.add_tokens(*args, **kwargs)
        return
    
    def train_(self):
        for module in self.children():
            for param in module.parameters():
                param.requires_grad = False
        
        return


if __name__ == '__main__':
    vision_text_embedding = VisionTextEmbedding('pretrained/stable-diffusion-v1-4/tokenizer', 'pretrained/stable-diffusion-v1-4/text_encoder')
    vision_text_embedding.add_tokens(['*'])
    # for i, k in enumerate(embedding_manager.string2token.keys()):
    #     embedding_manager.string2token[k] = torch.tensor(i)
    
    prompt = '*'
    image_embeds = torch.rand([1, 256, 768])
    vision_text_embedding(prompt, image_embeds)

    # text_embeds = embedding_manager(text_embeds, torch.tensor([0, 1]), image_embeds)
    # print(text_embeds)
