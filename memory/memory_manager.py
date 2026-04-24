import os
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import chromadb
import tiktoken
from llm_gateway import LLMGateway

class MemoryManager:
    """
    Enterprise Persistent Memory System.
    Handles semantic retrieval (ChromaDB), episodic logging (SQLite), 
    and working memory context window limits. Includes Multi-Session isolations.
    """
    def __init__(self, db_path: str = "data/memory/episodic.db", chroma_path: str = "data/memory/chroma", session_id: str = None):
        self.session_id = session_id or "default_session"
        self.db_path = Path(db_path)
        self.chroma_path = Path(chroma_path)
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Semantic Vector Store (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.knowledge_collection = self.chroma_client.get_or_create_collection(name="general_knowledge")
        
        # 2. Episodic Memory Logger (SQLite)
        self._init_sqlite()
        
        # 3. Working Context Manager
        self.working_context: List[Dict[str, Any]] = []
        self.max_working_tokens = 3000 # Configurable limit for Context Manager
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # 4. Entity Context Profiler
        self.profile_path = Path("data/user_profile.json")
        self.user_profile = self._load_profile()

    def _init_sqlite(self):
        """Initialise the temporal database schema mapping exact actions."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                task TEXT,
                action_tool TEXT,
                action_args TEXT,
                result TEXT,
                efe_score REAL
            )
        ''')
        
        # Retrofit legacy tables seamlessly
        try:
            cursor.execute('ALTER TABLE episodes ADD COLUMN session_id TEXT DEFAULT "default_session"')
        except sqlite3.OperationalError:
            pass
            
        self.conn.commit()
        
    def _load_profile(self) -> Dict[str, Any]:
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {"name": "User", "gmail": "", "preferences": {}}

    def save_profile_preference(self, key: str, value: Any):
        """Update company/user-specific metadata."""
        if "preferences" not in self.user_profile:
            self.user_profile["preferences"] = {}
        self.user_profile["preferences"][key] = value
        
        # Ensure main dir exists
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, 'w') as f:
            json.dump(self.user_profile, f, indent=4)

    # -----------------------------------------------------
    # 1. Semantic Vector Store Methods
    # -----------------------------------------------------
    def store_semantic_knowledge(self, document: str, metadata: Optional[Dict[str, Any]] = None):
        """Store knowledge in vector DB for fast, semantic retrieval of past knowledge."""
        doc_id = f"doc_{datetime.datetime.now().timestamp()}"
        
        # ChromaDB crashes if metadata is an empty dictionary {}
        safe_metadata = metadata if metadata else {"source": "system_memory"}
        
        self.knowledge_collection.add(
            documents=[document],
            metadatas=[safe_metadata],
            ids=[doc_id]
        )
        
    def retrieve_semantic_knowledge(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        # Handle if collection is empty
        if self.knowledge_collection.count() == 0:
            return []
            
        results = self.knowledge_collection.query(
            query_texts=[query],
            n_results=min(n_results, self.knowledge_collection.count())
        )
        
        docs = results.get('documents', [[]])[0]
        metas = results.get('metadatas', [[]])[0]
        
        return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]

    # -----------------------------------------------------
    # 2. Episodic Memory Logger Methods
    # -----------------------------------------------------
    def log_episode(self, task: str, tool: str, args: Dict, result: Any, efe_score: float = 0.0):
        """Map exact actions, states, and outcomes (past history) across sessions."""
        cursor = self.conn.cursor()
        
        # Handle complex results that aren't strings
        if isinstance(result, dict) or isinstance(result, list):
            result_str = json.dumps(result)
        else:
            result_str = str(result)
            
        cursor.execute(
            "INSERT INTO episodes (session_id, timestamp, task, action_tool, action_args, result, efe_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.session_id, datetime.datetime.now().isoformat(), task, tool, json.dumps(args), result_str, efe_score)
        )
        self.conn.commit()
        
    def get_recent_episodes(self, limit: int = 5) -> List[Dict]:
        """Fetch past execution history, strictly isolated to the multi-session ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT timestamp, task, action_tool, result FROM episodes WHERE session_id = ? ORDER BY id DESC LIMIT ?", 
            (self.session_id, limit,)
        )
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # -----------------------------------------------------
    # 3 & 5. Working Context Manager & Pruning
    # -----------------------------------------------------
    def add_working_context(self, entry: Dict[str, Any]):
        """Build a system to handle context window limits by summarising/pruning."""
        self.working_context.append(entry)
        self._prune_working_context()

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _prune_working_context(self):
        """
        Context Window Compression Algorithm: 
        Summarizing models to prevent LLM max-token crashes.
        """
        while self.working_context:
            total_text = json.dumps(self.working_context)
            if self._count_tokens(total_text) > self.max_working_tokens:
                print(f"[Memory Manager] Context exceeded {self.max_working_tokens} tokens! Executing Semantic Compression...")
                
                # Compress the oldest 3 items
                chunk_to_compress = self.working_context[:3]
                self.working_context = self.working_context[3:]
                
                gateway = LLMGateway()
                sys_prompt = "You are an AI Context Compressor. Summarize the following past execution steps concisely. Retain only critical system states, file paths, variables, and factual outcomes. Discard redundant conversational fluff."
                compressed_text = gateway.generate_completion(sys_prompt, json.dumps(chunk_to_compress))
                
                self.working_context.insert(0, {
                    "type": "compressed_history",
                    "summary": compressed_text
                })
            else:
                break
                
    def get_working_context(self) -> List[Dict]:
        return self.working_context

    def close(self):
        """Clean up connections."""
        self.conn.close()
