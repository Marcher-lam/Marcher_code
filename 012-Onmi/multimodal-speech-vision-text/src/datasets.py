import os
from torch.utils.data import Dataset
from .audio_utils import load_audio,compute_mel_spec
from .image_utils import load_image
from .text_utils import tokenize_text,get_tokenizer

class MultiModalDataset(Dataset):
    def __init__(self,data_list,tokenizer,config):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data_list)
    def __getitem__(self,idx):
        item = self.data_list[idx]
        audio = None
        img = None
        text = None

        if item.get("audio_path"):
            waveform = load_audio(item["audio_path"],target_sr=self.config['audio_sample_rate'])
            mel = compute_mel_spec(waveform,sr = self.config['audio_sample_rate'],n_mels = self.config['n_mels'])
            audio = mel

        if item.get("img_path"):
            img = load_image(item["img_path"])

        if item.get("text"):
            input_ids,attn_mask = tokenize_text(self.tokenizer,item["text"],self.config['max_text_len'])
            text = {
                "input_ids":input_ids,
                "attention_mask":attn_mask,
                "labels":input_ids
            }
            return {
                "audio":audio,
                "img":img,
                "text":text
            }