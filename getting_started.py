from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
import time
import json
import os



from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import torch


def get_process_id():
    return os.getpid()

EMBED_MODEL_ID = "allenai/specter2_base"  # tokenizer lives on base, not the adapter
MAX_TOKENS = 512   # SPECTER2's max sequence length 

print(get_process_id())                       
time.sleep(10)

def build_doc_converter() -> DocumentConverter:
    '''
    Not running OCR
    '''
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
    )
    return converter


def build_chunker() -> HybridChunker:
     '''
     Chunker is intelligent here, it enforces embedding model's token limit on top.

     It walks the body tree (the hierarchical structure from DoclingDocument), groups nodes that belong together logically, 
     enforces token limit by splitting things that are too big and merging things that are too small within the same section, 
     and attaches the heading path to each resulting chunk. 
     '''
     tokenizer = HuggingFaceTokenizer(
          tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
          max_tokens=MAX_TOKENS
     )
     return HybridChunker(tokenizer=tokenizer, merge_peers=True)

converter = build_doc_converter()
chunker = build_chunker()


class PDFProcessingError(Exception):
    """
    Raised when a PDF cannot be processed end-to-end.
 
    Wraps lower-level library errors (e.g. docling conversion failures)
    into a single domain-specific type the batch layer can catch cleanly.
    """


def process_pdf(file_path: str, converter: DocumentConverter, chunker: HybridChunker) -> None:
    """
    Raises: PDFProcessingError
    """

    print(f"Processing started for: {file_path}")
    file_path_jsonl = "./academic_papers_jsonl/" + file_path.split("/")[-1].replace(".pdf", ".jsonl")

    try:
        file_result = converter.convert(file_path)
    except Exception as e:
        raise PDFProcessingError(f"Failed to process PDF {file_path}") from e
    
    doc = file_result.document

    with open(file_path_jsonl, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
            first_item = chunk.meta.doc_items[0] if chunk.meta.doc_items else None
            page_no = first_item.prov[0].page_no if (first_item and first_item.prov) else None

            record = {
                    "doc_name": file_path.split("/")[-1],
                    "chunk_id": i,
                    "text": chunk.text,
                    "contextualized_text": chunker.contextualize(chunk=chunk),
                    "headings": getattr(chunk.meta, "headings", None),
                    "page_no": page_no,
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Processing completed for: {file_path_jsonl}")
    return (file_path_jsonl, True)

print(get_process_id())

list_file_path = glob("./academic_papers/*")

start_time = time.time()
for file_path in list_file_path:
    process_pdf(file_path=file_path, converter=converter, chunker=chunker)

print(f"Total time taken for processing: {time.time() - start_time}")