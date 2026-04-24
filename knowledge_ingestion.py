import os
import json
import ast
import csv
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    Observer = None
    FileSystemEventHandler = object

try:
    import pypdf
except ImportError:
    pypdf = None

class KnowledgeIngestor:
    """
    Knowledge Ingestion System (Phase 3).
    Parses Code, PDF, CSV, and Markdown into intelligent chunks,
    then continuously syncs them to the Semantic Memory Vector DB.
    """
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.chunk_size = 1000  # Default character chunk size
        self.observer = None

    def ingest_file(self, filepath: str):
        path = Path(filepath)
        if not path.exists():
            return
        
        ext = path.suffix.lower()
        if ext == '.pdf':
            self._ingest_pdf(path)
        elif ext in ['.md', '.txt']:
            self._ingest_markdown(path)
        elif ext == '.csv':
            self._ingest_csv(path)
        elif ext == '.py':
            self._ingest_python(path)
        else:
            self._ingest_generic(path)
            
    def _ingest_pdf(self, path: Path):
        print(f"[Ingestor] Parsing PDF: {path.name}")
        if not pypdf:
            print("[Ingestor] Warning: pypdf not installed. Skipping PDF ingestion.")
            return
        try:
            with open(path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                self._chunk_and_store(text, {"source": str(path), "type": "pdf"})
        except Exception as e:
            print(f"[Ingestor] Error reading PDF: {e}")

    def _ingest_python(self, path: Path):
        print(f"[Ingestor] Intelligent AST Parsing Python Code: {path.name}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Intelligent chunking via AST to keep classes/functions logically bound
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chunk = ast.get_source_segment(content, node)
                    if chunk:
                        self.memory.store_semantic_knowledge(
                            chunk, 
                            {"source": str(path), "type": "code", "entity_name": node.name}
                        )
            
            # Store a generic chunk for file-level module docstrings/variables
            self.memory.store_semantic_knowledge(
                content[:self.chunk_size], 
                {"source": str(path), "type": "code_summary"}
            )
        except Exception:
            # Fallback for syntax errors
            print(f"[Ingestor] AST Parsing failed for {path.name}, falling back to generic chunking.")
            self._ingest_generic(path)

    def _ingest_markdown(self, path: Path):
        print(f"[Ingestor] Chunking Markdown: {path.name}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Intelligent chunking by heading layout
            chunks = content.split('\n## ')
            for i, chunk in enumerate(chunks):
                text = f"## {chunk}" if i > 0 else chunk
                self._chunk_and_store(text, {"source": str(path), "type": "markdown"})
        except Exception:
            self._ingest_generic(path)

    def _ingest_csv(self, path: Path):
        print(f"[Ingestor] Parsing CSV: {path.name}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                # Compress into 10-row json blocks for the Vector DB
                for i in range(0, len(rows), 10):
                    chunk_text = json.dumps(rows[i:i+10])
                    self._chunk_and_store(chunk_text, {"source": str(path), "type": "csv_rows", "row_start": i})
        except Exception as e:
            print(f"[Ingestor] Error reading CSV: {e}")

    def _ingest_generic(self, path: Path):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            self._chunk_and_store(content, {"source": str(path), "type": "text"})
        except:
            pass

    def _chunk_and_store(self, text: str, metadata: dict):
        """Naive fixed-size chunking fallback."""
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i+self.chunk_size]
            if len(chunk.strip()) > 10:
                self.memory.store_semantic_knowledge(chunk, metadata)

    def watch_directory(self, dir_path: str):
        """Continuous Sync: file watchers to update the index dynamically."""
        if Observer is None:
            print("[Ingestor] Warning: watchdog library not installed. Continuous sync disabled.")
            return

        class IngestEventHandler(FileSystemEventHandler):
            def __init__(self, ingestor):
                self.ingestor = ingestor
            
            def on_modified(self, event):
                if not event.is_directory:
                    self.ingestor.ingest_file(event.src_path)

            def on_created(self, event):
                if not event.is_directory:
                    self.ingestor.ingest_file(event.src_path)

        path = Path(dir_path)
        if not path.exists():
            return
            
        print(f"[KnowledgeIngestor] Starting background continuous sync on '{dir_path}'...")
        self.observer = Observer()
        self.observer.schedule(IngestEventHandler(self), path=str(path), recursive=True)
        self.observer.start()
        
    def stop_watching(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
