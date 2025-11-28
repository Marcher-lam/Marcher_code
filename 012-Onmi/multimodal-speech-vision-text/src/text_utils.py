from transformers import AutoTokenizer


def get_tokenizer(model_name="bert-base-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_text(tokenizer,text,max_len=512):
    d = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return d.input_ids.squeeze(0),d.attention_mask.squeeze(0)