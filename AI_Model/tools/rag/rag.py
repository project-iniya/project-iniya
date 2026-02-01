"""
Improved Retrieval-Augmented Generation (RAG) System
with enhanced chunking, error handling, and metadata management.

Available Functions:
- load_pdf(pdf_path, chunk_size=400, force_reload=False, use_ocr=False)
- load_text(text, doc_id, chunk_size=400)
- load_image(image_path, doc_id=None)
- load_docx(docx_path, doc_id=None)
- load_csv(csv_path, doc_id=None)
- load_excel(excel_path, doc_id=None)
- load_file(file_path, doc_id=None, use_ocr=False)
- retrieve(query, top_k=5)
- get_context(query, top_k=5, include_metadata=True)
- ask(query, top_k=5, model=CHAT_MODEL)

"""

import ollama
import chromadb
import PyPDF2
import os
import sys
import pytesseract
from PIL import Image
from docx import Document
import pandas as pd
from pdf2image import convert_from_path
import re
from typing import List, Dict, Optional

# Configuration
EMBED_MODEL = "mxbai-embed-large"
DB_PATH = "AI_Model/memory/rag_memory"

# Chunk sizes - REDUCED to fit within model limits
DEFAULT_CHUNK_SIZE = 400  # Reduced from 600
MAX_CHUNK_SIZE = 200  # Fallback if 400 fails

# Optional: Set your poppler path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
POPPLER_PATH = os.path.join(project_root, "poppler", "Library", "bin")
sys.path.insert(0, project_root)

from AI_Model.config import DEFAULT_MODEL as CHAT_MODEL  # Ensure this path is correct


class ImprovedRAGSystem:
    def __init__(self, collection_name="documents", persist_path=DB_PATH):
        """Initialize RAG system with persistent storage"""
        os.makedirs(persist_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection_name = collection_name
        
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✓ Loaded existing collection: {collection_name}")
            print(f"  Contains {self.collection.count()} chunks")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"✓ Created new collection: {collection_name}")
    
    def _smart_chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = 50) -> List[str]:
        """
        Intelligently chunk text by sentences to preserve semantic meaning.
        Adjusted for embedding model limits.
        """
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_words > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap from previous chunk
                overlap_words = []
                overlap_count = 0
                for s in reversed(current_chunk):
                    words = len(s.split())
                    if overlap_count + words <= overlap:
                        overlap_words.insert(0, s)
                        overlap_count += words
                    else:
                        break
                
                current_chunk = overlap_words
                current_length = overlap_count
            
            current_chunk.append(sentence)
            current_length += sentence_words
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Filter out tiny chunks
        return [c for c in chunks if len(c.split()) > 10]
    
    def _split_chunk_further(self, chunk: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
        """Split a chunk that's too large into smaller pieces"""
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        
        sub_chunks = []
        current = []
        current_len = 0
        
        for sentence in sentences:
            s_len = len(sentence.split())
            
            if current_len + s_len > max_size and current:
                sub_chunks.append(' '.join(current))
                current = [sentence]
                current_len = s_len
            else:
                current.append(sentence)
                current_len += s_len
        
        if current:
            sub_chunks.append(' '.join(current))
        
        return sub_chunks
    
    def _embed_chunk_with_retry(self, chunk: str, chunk_id: str) -> Optional[List[float]]:
        """Try to embed a chunk, splitting it if it's too large"""
        try:
            # Try normal embedding
            res = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
            return res['embedding']
        
        except Exception as e:
            if "exceeds the context length" in str(e).lower():
                print(f"⚠ Chunk too large, splitting into smaller pieces...")
                
                # Split the chunk
                sub_chunks = self._split_chunk_further(chunk, max_size=MAX_CHUNK_SIZE)
                
                # Try to embed each sub-chunk
                embeddings = []
                for i, sub_chunk in enumerate(sub_chunks):
                    try:
                        res = ollama.embeddings(model=EMBED_MODEL, prompt=sub_chunk)
                        embeddings.append(res['embedding'])
                    except Exception as e2:
                        print(f"⚠ Even smaller chunk failed: {e2}")
                        # Last resort: take first 100 words
                        words = sub_chunk.split()[:100]
                        mini_chunk = ' '.join(words)
                        try:
                            res = ollama.embeddings(model=EMBED_MODEL, prompt=mini_chunk)
                            embeddings.append(res['embedding'])
                        except:
                            print(f"✗ Could not embed chunk at all, skipping")
                            return None
                
                # Average the embeddings
                if embeddings:
                    avg_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
                    return avg_embedding
                
                return None
            else:
                print(f"✗ Embedding error: {e}")
                return None
    
    def _extract_pdf_text(self, pdf_path: str, use_ocr: bool = False) -> tuple[str, Dict]:
        """
        Extract text from PDF with metadata about pages.
        Returns (text, metadata_dict)
        """
        text = ""
        page_metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                print(f"PDF has {total_pages} pages")
                
                # Try regular extraction first
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            page_start = len(text)
                            text += f"\n{page_text}"
                            page_end = len(text)
                            page_metadata[page_num + 1] = (page_start, page_end)
                    except Exception as e:
                        print(f"⚠ Error extracting page {page_num + 1}: {e}")
                
                # Use OCR if needed
                if not text or len(text.strip()) < 100 or use_ocr:
                    print("Using OCR for text extraction...")
                    text, page_metadata = self._extract_pdf_with_ocr(pdf_path)
                else:
                    print(f"Extracted text from {total_pages} pages")
        
        except Exception as e:
            print(f"✗ Error reading PDF: {e}")
            raise
        
        return text.strip(), page_metadata
    
    def _extract_pdf_with_ocr(self, pdf_path: str) -> tuple[str, Dict]:
        """Extract text using OCR and return page metadata"""
        try:
            print("Converting PDF to images for OCR...")
            
            if POPPLER_PATH and os.path.exists(POPPLER_PATH):
                images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
            else:
                images = convert_from_path(pdf_path)
            
            text = ""
            page_metadata = {}
            
            for page_num, image in enumerate(images):
                print(f"OCR processing page {page_num + 1}/{len(images)}...")
                
                page_start = len(text)
                page_text = pytesseract.image_to_string(image)
                text += f"\n{page_text}"
                page_end = len(text)
                
                page_metadata[page_num + 1] = (page_start, page_end)
            
            print(f"✓ OCR completed for {len(images)} pages")
            return text, page_metadata
        
        except Exception as e:
            print(f"✗ Error during OCR: {e}")
            return "", {}
    
    def _get_page_number(self, chunk_start_pos: int, page_metadata: Dict) -> int:
        """Determine which page a chunk belongs to based on character position"""
        for page_num, (start, end) in page_metadata.items():
            if start <= chunk_start_pos <= end:
                return page_num
        return 1  # Default to page 1 if not found
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if document is already loaded"""
        try:
            results = self.collection.get(where={"source": doc_id})
            return len(results['ids']) > 0
        except:
            return False
    
    def load_pdf(self, pdf_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, force_reload: bool = False, use_ocr: bool = False):
        """
        Load and index a PDF file with improved chunking and metadata
        """
        doc_id = os.path.basename(pdf_path)
        
        if not force_reload and self.document_exists(doc_id):
            print(f"⚠ Document '{doc_id}' already loaded. Use force_reload=True to reload.")
            return
        
        print(f"Loading PDF: {pdf_path}")
        
        try:
            # Extract text with page metadata
            text, page_metadata = self._extract_pdf_text(pdf_path, use_ocr)
            
            if not text or len(text.strip()) < 10:
                print("⚠ No text extracted from PDF")
                return
            
            # Smart chunking
            chunks = self._smart_chunk_text(text, chunk_size)
            print(f"Created {len(chunks)} chunks with smart sentence-based splitting")
            
            # Track position for page numbers
            current_pos = 0
            successful_chunks = 0
            
            # Generate embeddings and store
            for i, chunk in enumerate(chunks):
                # Find chunk position in original text
                chunk_pos = text.find(chunk[:50])  # Find by first 50 chars
                if chunk_pos == -1:
                    chunk_pos = current_pos
                
                page_num = self._get_page_number(chunk_pos, page_metadata)
                current_pos = chunk_pos + len(chunk)
                
                # Try to embed with auto-retry/split
                embedding = self._embed_chunk_with_retry(chunk, f"{doc_id}_chunk_{i}")
                
                if embedding is None:
                    print(f"⚠ Skipping chunk {i} - could not generate embedding")
                    continue
                
                try:
                    # Store with rich metadata
                    self.collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{
                            "source": doc_id,
                            "chunk_index": i,
                            "type": "pdf",
                            "page": page_num,
                            "char_count": len(chunk),
                            "word_count": len(chunk.split())
                        }],
                        ids=[f"{doc_id}_chunk_{i}"]
                    )
                    successful_chunks += 1
                    
                    if (successful_chunks) % 10 == 0:
                        print(f"✓ Processed {successful_chunks}/{len(chunks)} chunks")
                
                except Exception as e:
                    print(f"⚠ Error storing chunk {i}: {e}")
                    continue
            
            print(f"✓ PDF loaded: {successful_chunks}/{len(chunks)} chunks successfully indexed")
        
        except Exception as e:
            print(f"✗ Error loading PDF: {e}")
            raise
    
    def load_text(self, text: str, doc_id: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """Load plain text with smart chunking"""
        if self.document_exists(doc_id):
            print(f"⚠ Document '{doc_id}' already loaded")
            return
        
        print(f"Loading text: {doc_id}")
        
        chunks = self._smart_chunk_text(text, chunk_size)
        print(f"Created {len(chunks)} chunks")
        
        successful_chunks = 0
        for i, chunk in enumerate(chunks):
            embedding = self._embed_chunk_with_retry(chunk, f"{doc_id}_chunk_{i}")
            
            if embedding is None:
                continue
            
            try:
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": doc_id,
                        "chunk_index": i,
                        "type": "text",
                        "word_count": len(chunk.split())
                    }],
                    ids=[f"{doc_id}_chunk_{i}"]
                )
                successful_chunks += 1
            except Exception as e:
                print(f"⚠ Error storing chunk {i}: {e}")
                continue
        
        print(f"✓ Text loaded: {successful_chunks}/{len(chunks)} chunks")
    
    def load_image(self, image_path: str, doc_id: Optional[str] = None):
        """Extract text from image using OCR"""
        try:
            if doc_id is None:
                doc_id = os.path.basename(image_path)
            
            if self.document_exists(doc_id):
                print(f"⚠ Document '{doc_id}' already loaded")
                return
            
            print(f"Loading image: {image_path}")
            
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            
            if not text or len(text.strip()) < 10:
                print("⚠ No text extracted from image")
                return
            
            self.load_text(text, doc_id=doc_id)
            print("✓ Image loaded successfully!")
        
        except Exception as e:
            print(f"✗ Error loading image: {e}")
    
    def load_docx(self, docx_path: str, doc_id: Optional[str] = None):
        """Load Word document with paragraph preservation"""
        try:
            if doc_id is None:
                doc_id = os.path.basename(docx_path)
            
            if self.document_exists(doc_id):
                print(f"⚠ Document '{doc_id}' already loaded")
                return
            
            print(f"Loading Word document: {docx_path}")
            
            doc = Document(docx_path)
            text = '\n\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            
            if not text or len(text.strip()) < 10:
                print("⚠ No text extracted from Word document")
                return
            
            self.load_text(text, doc_id=doc_id)
            print("✓ Word document loaded successfully!")
        
        except Exception as e:
            print(f"✗ Error loading Word document: {e}")
    
    def load_csv(self, csv_path: str, doc_id: Optional[str] = None):
        """Load CSV with better formatting"""
        try:
            if doc_id is None:
                doc_id = os.path.basename(csv_path)
            
            if self.document_exists(doc_id):
                print(f"⚠ Document '{doc_id}' already loaded")
                return
            
            print(f"Loading CSV file: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            # Create a more structured text representation
            text = f"CSV File: {doc_id}\n"
            text += f"Columns: {', '.join(df.columns)}\n"
            text += f"Rows: {len(df)}\n\n"
            
            # Add row-by-row data for better chunking
            for idx, row in df.iterrows():
                if idx > 1000:  # Limit rows
                    text += f"... {len(df) - idx} more rows omitted ...\n"
                    break
                row_text = f"Row {idx + 1}: " + ", ".join([f"{col}={val}" for col, val in row.items()])
                text += row_text + "\n"
            
            self.load_text(text, doc_id=doc_id)
            print("✓ CSV file loaded successfully!")
        
        except Exception as e:
            print(f"✗ Error loading CSV file: {e}")
    
    def load_excel(self, excel_path: str, doc_id: Optional[str] = None):
        """Load Excel with sheet-aware processing"""
        try:
            if doc_id is None:
                doc_id = os.path.basename(excel_path)
            
            if self.document_exists(doc_id):
                print(f"⚠ Document '{doc_id}' already loaded")
                return
            
            print(f"Loading Excel file: {excel_path}")
            
            excel_file = pd.ExcelFile(excel_path)
            all_text = [f"Excel File: {doc_id}\n"]
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                sheet_text = f"\n=== Sheet: {sheet_name} ===\n"
                sheet_text += f"Columns: {', '.join(df.columns)}\n"
                sheet_text += f"Rows: {len(df)}\n\n"
                
                # Add data row by row
                for idx, row in df.iterrows():
                    if idx > 500:  # Limit per sheet
                        sheet_text += f"... {len(df) - idx} more rows omitted ...\n"
                        break
                    row_text = f"Row {idx + 1}: " + ", ".join([f"{col}={val}" for col, val in row.items()])
                    sheet_text += row_text + "\n"
                
                all_text.append(sheet_text)
            
            text = '\n'.join(all_text)
            self.load_text(text, doc_id=doc_id)
            print("✓ Excel file loaded successfully!")
        
        except Exception as e:
            print(f"✗ Error loading Excel file: {e}")
    
    def load_file(self, file_path: str, doc_id: Optional[str] = None, use_ocr: bool = False):
        """Auto-detect and load any supported file type"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            self.load_pdf(file_path, use_ocr=use_ocr)
        elif ext == '.docx':
            self.load_docx(file_path, doc_id)
        elif ext in ['.xlsx', '.xls']:
            self.load_excel(file_path, doc_id)
        elif ext == '.csv':
            self.load_csv(file_path, doc_id)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            self.load_image(file_path, doc_id)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.load_text(text, doc_id or os.path.basename(file_path))
        else:
            print(f"✗ Unsupported file type: {ext}")
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """Retrieve most relevant chunks with better ranking"""
        # Generate query embedding
        q_res = ollama.embeddings(model=EMBED_MODEL, prompt=query)
        
        # Retrieve more than needed for re-ranking
        results = self.collection.query(
            query_embeddings=[q_res['embedding']],
            n_results=min(top_k * 2, 10)  # Get extra for filtering
        )
        
        if not results['documents'] or not results['documents'][0]:
            return {'documents': [], 'metadatas': [], 'relevance_scores': []}
        
        # Return top results with metadata
        docs = results['documents'][0][:top_k]
        metas = results['metadatas'][0][:top_k] if results['metadatas'] else []
        distances = results['distances'][0][:top_k] if results.get('distances') else []
        
        return {
            'documents': docs,
            'metadatas': metas,
            'relevance_scores': [1 / (1 + d) for d in distances]  # Convert distance to relevance
        }
    
    def get_context(self, query: str, top_k: int = 5, include_metadata: bool = True) -> str:
        """Get formatted context string with optional metadata"""
        results = self.retrieve(query, top_k)
        
        if not results['documents']:
            return "No relevant information found."
        
        context_parts = []
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            if include_metadata and meta:
                source = meta.get('source', 'Unknown')
                page = meta.get('page', '?')
                header = f"[Source: {source}, Page: {page}]"
                context_parts.append(f"{header}\n{doc}")
            else:
                context_parts.append(doc)
        
        return "\n\n---\n\n".join(context_parts)
    
    def ask(self, query: str, top_k: int = 5, model: str = CHAT_MODEL) -> str:
        """
        Ask a question and get an AI-generated answer based on retrieved context.
        This is the main method you'd use for RAG Q&A.
        """
        # Get relevant context
        context = self.get_context(query, top_k, include_metadata=True)
        
        if context == "No relevant information found.":
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Create prompt for the LLM
        prompt = f"""Based on the following context from documents, answer the question.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        try:
            response = ollama.generate(model=model, prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Error generating response: {e}"
    
    def list_documents(self) -> List[str]:
        """List all loaded documents"""
        try:
            all_data = self.collection.get()
            sources = set()
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    if 'source' in metadata:
                        sources.add(metadata['source'])
            return sorted(list(sources))
        except:
            return []
    
    def delete_document(self, doc_id: str):
        """Delete a specific document"""
        try:
            self.collection.delete(where={"source": doc_id})
            print(f"✓ Deleted document: {doc_id}")
        except Exception as e:
            print(f"✗ Error deleting document: {e}")
    
    def clear_collection(self):
        """Clear all documents from collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)
        print("✓ Collection cleared")
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        count = self.collection.count()
        docs = self.list_documents()
        return {
            'total_chunks': count,
            'documents': docs,
            'num_documents': len(docs)
        }


# Example usage
if __name__ == "__main__":
    # Initialize the system
    rag = ImprovedRAGSystem()

    import time
    # Load a PDF (with automatic fallback to OCR if needed)
    start = time.perf_counter()
    rag.load_pdf("sample.pdf", force_reload=True)
    rag.load_docx("sample.docx")
    # Ask questions
    answer = rag.ask("What is the main topic of the document?")
    print(f"Answer: {answer}")
    
    # Get raw context without AI response
    context = rag.get_context("main topic", top_k=3)
    print(f"Context: {context}")

    elapsed = time.perf_counter() - start

    # Check stats
    stats = rag.get_stats()
    print(f"\nRAG System Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Files: {stats['documents']}")