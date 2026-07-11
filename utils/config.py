from pathlib import Path

EMBED_MODEL_ID = "allenai/specter2_base"
EMBED_MODEL_MAX_TOKENS = 512
CHUNK_BATCH_SIZE = 3

# Paths relative to this file's location, works regardless of where code is executed from
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent
JSONL_PATH = str(PROJECT_ROOT) + "/" + "academic_papers_jsonl"