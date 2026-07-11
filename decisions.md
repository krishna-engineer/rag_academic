## Corpus
- ~15 arXiv papers across two deliberately separate clusters (KV cache / LLM inference optimization + LLM text detection). 
- The gap between clusters is intentional — it tests whether retrieval correctly avoids irrelevant content, not just whether it finds relevant content.

## Embedding model: allenai/specter2
- Domain-fit: trained on scientific/academic paper data via citation graphs, so it handles dense technical language better than general-purpose embedders. 

## LLM: gemma4:12b via Ollama (local)
- At 7.6GB it fits comfortably in 16GB unified memory on the M4 with headroom for OS/app
- larger variants (26b/31b) would thrash. 
- Local chosen to avoid API cost and keep data on-device.

## Extraction: Docling (MinerU evaluated and rejected)
- Chose Docling for structure-aware extraction to JSONL. Found that Docling outout was poor for table dat extraction.
- Evaluated MinerU as a stronger alternative for table extraction, but rejected it — MinerU needs ~8–16GB VRAM, and on a 16GB MacBook Air with other apps (Chrome, etc.) consuming RAM, only ~1–2GB is realistically free.
- Accepted Docling's imperfect table ordering as out-of-scope for a baseline, since research-paper answerable content lives mostly in prose and the LLM can recover from mildly disordered table text.

## Embedding field: contextualized_text (heading prepended)
- Embedding the heading-enriched text gives retrieval both local content and structural context. Chose this over raw text despite the cost noted below.

## Truncation over splitting (deliberate)
- The 512 cap is enforced on raw text, but headings are prepended after, so contextualized_text can run ~7–11 tokens over 512 and specter2 silently truncates the tail.
- Accepted this small, front-safe loss rather than lowering the cap and fragmenting coherent sections into weaker chunks.

## 2 adapters approach
- embed source doc chunks with the proximity adapter (allenai/specter2, also called proximity/retrieval)
- embed the user query at retrieval time with the adhoc_query adapter (allenai/specter2_adhoc_query).

## Unqiue chunk-id
- Currently going ahead with "filename_chunkid" as unique id of any chunk across all files
- But above approach will also consider files where filename is different but content is 100% same. Ideally only one of those files should be considered.
- At enterprise scale, dedup should happen at document ingestion stage (by doing hash of entire file content) i.e. "before" text extraction+chunking level.

## Chunk embedding and Vector DB
- For this build, batched embeddings will be stored directly in Vector DB without with retry + dead letter on failures
- At enterprise scale, I'd add a DB-agnostic parquet embedding store as the source of truth, which will make Vector DB swappable + rebuilding will not be expensive 
