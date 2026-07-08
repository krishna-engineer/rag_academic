import config
from glob import glob
import json

from transformers import AutoTokenizer

EMBED_MODEL_ID = config.EMBED_MODEL_ID
EMBED_MODEL_MAX_TOKENS = config.EMBED_MODEL_MAX_TOKENS
# TEXT_FIELD = "contextualized_text"
TEXT_FIELD = "text"

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

jsonl_files = glob("../academic_papers_jsonl/*.jsonl")
number_of_files_to_process = 2

for idx in range(0, number_of_files_to_process):
    jsonl_file = jsonl_files[idx]
    print(f"Processing file - {jsonl_file}")
    with open(jsonl_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            text_dict = json.loads(line)
            text = text_dict[TEXT_FIELD]
            token_count = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            
            if token_count >= EMBED_MODEL_MAX_TOKENS:
                print(token_count)
    