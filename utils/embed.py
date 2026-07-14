from adapters import AutoAdapterModel
from transformers import AutoTokenizer
import torch
import numpy as np

from utils import config

EMBED_MODEL_ID = config.EMBED_MODEL_ID
EMBED_ADAPTER_FOR_SOURCE = config.EMBED_ADAPTER_FOR_SOURCE


'''
insight: 
    - tokenizer adds [CLS] token at position 0, and embedding of this token has condensed summary
         of all tokens in that chunk (attention)
    - embeddings = output.last_hidden_state[:, 0, :] => batch_size(32), 1st CLS token, 768 token vector length 
'''
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
model = AutoAdapterModel.from_pretrained(EMBED_MODEL_ID)
model.load_adapter(EMBED_ADAPTER_FOR_SOURCE, source="hf", load_as="proximity", set_active=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

#insight: switches the model into inference mode by setting an internal self.training flag to False 
model.eval()


#to-do: convert this to class
def embed_batch(text_batch: list[str], max_token_length: int = 512) -> np.ndarray:
    inputs = tokenizer(text_batch, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=max_token_length)
    
    #insight: model and inputs must be on the same device, or you get a runtime error.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)

    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()