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


converter = None
chunker = None
# max_workers = max(1, os.cpu_count() - 2) # Leave 2 core free for the OS/other processes
max_workers = 3

def _init_worker():
    """
    Runs once per worker process when the pool starts.
    This is where the expensive model loading happens — exactly once
    per process, not once per file.
    """
    global converter, chunker
    torch.set_num_threads(1)
    converter = build_doc_converter()
    chunker = build_chunker()


def _process_pdf_worker(file_path: str):
    """
    Thin wrapper that uses the worker-local converter/chunker.
    Must be top-level (not a lambda, not a nested function) so it's picklable.
    """
    try:
        result = process_pdf(file_path=file_path, converter=converter, chunker=chunker)
        file_path_jsonl, is_success = result
        return (file_path, is_success, "")
    except Exception as e:
        # Don't let one bad PDF kill the whole batch — capture and report
        return (file_path, False, str(e))
    

def run_parallel(file_paths, max_workers=4):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_process_pdf_worker, fp): fp for fp in file_paths}

        for future in as_completed(futures):
            file_path, is_success, error_msg = future.result()
            if is_success is False:
                print(f"FAILED: {file_path} — {error_msg}")
            else:
                print(f"DONE: {file_path}")
    return results

if __name__ == "__main__":
    list_file_path = glob("./academic_papers/*")
    print(f"Count of files to process: {len(list_file_path)}")
    print(f"Number of workers used: {max_workers}")


    print(get_process_id())                       
    time.sleep(10)

    start_time = time.time()
    run_parallel(list_file_path, max_workers=max_workers)
    print(f"Total time taken for processing: {time.time() - start_time}")

