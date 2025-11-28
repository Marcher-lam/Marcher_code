import torch
import torch.nn as nn
from transformers import Wav2Vec2Model,ViTModel,GPT2LMHeadModel

class MultiModalFusionModel(nn.Module):
    def __init__(self,audio_model,image_model,text_model,hidden_dim=768,num_fusion_layers=2):
        super(MultiModalFusionModel,self).__init__()
        self.wav_encoder = Wav2Vec2Model.from_pretrained(audio_model)
        self.vit_encoder = ViTModel.from_pretrained(image_model)
        self.gpt_encoder = GPT2LMHeadModel.from_pretrained(text_model)

        d_a = self.wav_encoder.config.hidden_size
        d_i = self.vit_encoder.config.hidden_size
        self.proj_audio = nn.Linear(d_a,hidden_dim)
        self.proj_image = nn.Linear(d_i,hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=8)
        self.fusion_encoder = nn.TransformerEncoderLayer(encoder_layer,num_layers = num_fusion_layers)

        d_t = self.gpt_encoder.config.hidden_size
        self.to_gpt = nn.Linear(hidden_dim,d_t)

    def forward(self,audio=None,image=None,text=None,text_input_ids=None,text_attention_mask=None):
        batch_size = None

        if audio is not None:
            wav_out = self.wav_encoder(audio).lastt_hidden_state
            wav_proj = self.proj_audio(wav_out)
        else:
            wav_proj = None

        if image is not None:
            vit_out = self.vit_model(pixel_values=image).lastt_hidden_state
            img_proj = self.proj_image(vit_out)
        else:
            img_proj = None

        seqs = []
        if wav_proj is not None:
            seqs.append(wav_proj)
        if img_proj is not None:
            seqs.append(img_proj)
        if len(seqs) == 0:
            raise ValueError("No input provided")
        fused_seq = torch.cat(seqs, dim=-1)
        fused_seq_t = fused_seq.transpose(0,1)
        fused_enc = self.fusion_encoder(fused_seq_t)
        fused_enc = fused_enc.transpose(0,1)

        gpt_embeds = self.to_gpt(fused_enc)
        if text_input_ids is not None:
            prompt_embeds = self.gpt.transformer.wte(text_input_ids)
            all_embeds = torch.cat([gpt_embeds,prompt_embeds], dim=1)
            outputs = self.gpt(inputs_embeds=all_embeds,attention_mask=None)
        else:
            outputs = self.gpt(inputs_embeds=gpt_embeds,attention_mask=None)
        return outputs