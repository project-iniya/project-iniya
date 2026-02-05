# rag.py
import ollama
import chromadb
import PyPDF2
import os, sys
import pytesseract
from PIL import Image
from docx import Document
import pandas as pd
from pdf2image import convert_from_path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from AI_Model.log import log
from AI_Model.config import CURRENT_CHAT_ID

EMBED_MODEL = "mxbai-embed-large"
DB_PATH = f"AI_Model/memory/{CURRENT_CHAT_ID}/rag_memory"
POPPLER_PATH = os.path.join(project_root, "poppler", "Library", "bin")

class RAGSystem:
    def __init__(self, collection_name="documents", persist_path=DB_PATH):
        """Initialize RAG system with persistent storage"""
        # Store log function
        self.log = log
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_path, exist_ok=True)
        
        # Use persistent client
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection_name = collection_name
        
        # Try to get existing collection or create new one
        try:
            self.collection = self.client.get_collection(collection_name)
            self.log(f"✓ Loaded existing collection: {collection_name}", "RAG")
            self.log(f"  Contains {self.collection.count()} chunks", "RAG")
        except:
            self.collection = self.client.create_collection(collection_name)
            self.log(f"✓ Created new collection: {collection_name}", "RAG")
    
    def _extract_pdf_text(self, pdf_path, use_ocr=False):
        """Extract text from PDF file with OCR fallback"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                self.log(f"PDF has {total_pages} pages", "RAG")
                
                # First try regular text extraction
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        self.log(f"⚠ Error extracting page {page_num + 1}: {e}", "RAG")
                        continue
                
                # If no text extracted or very little text, use OCR
                if not text or len(text.strip()) < 100 or use_ocr:
                    self.log("Text extraction yielded little/no text. Using OCR...", "RAG")
                    text = self._extract_pdf_with_ocr(pdf_path)
                else:
                    self.log(f"Extracted text from {total_pages} pages", "RAG")
                    
        except Exception as e:
            self.log(f"✗ Error reading PDF: {e}", "RAG")
            raise
        
        return text.strip()
    
    def _extract_pdf_with_ocr(self, pdf_path):
        """Extract text from PDF using OCR"""
        try:
            self.log("Converting PDF to images for OCR...", "RAG")
            
            # Convert PDF to images with poppler path
            if POPPLER_PATH and os.path.exists(POPPLER_PATH):
                images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
            else:
                # Try without poppler_path (if it's in system PATH)
                try:
                    images = convert_from_path(pdf_path)
                except Exception as e:
                    self.log(f"✗ Poppler not found. Please set POPPLER_PATH in rag.py", "RAG")
                    self.log(f"Error: {e}", "RAG")
                    return ""
            
            text = ""
            total_pages = len(images)
            
            for page_num, image in enumerate(images):
                self.log(f"Processing page {page_num + 1}/{total_pages} with OCR...", "RAG")
                
                # Extract text using OCR
                page_text = pytesseract.image_to_string(image)
                
                if page_text:
                    text += f"\n--- Page {page_num + 1} (OCR) ---\n{page_text}"
                
                if (page_num + 1) % 5 == 0:
                    self.log(f"OCR processed {page_num + 1}/{total_pages} pages", "RAG")
            
            self.log(f"✓ OCR completed for all {total_pages} pages", "RAG")
            return text
            
        except Exception as e:
            self.log(f"✗ Error during OCR: {e}", "RAG")
            return ""
    
    def _chunk_text(self, text, chunk_size=300, overlap=50):
        """Split text into overlapping chunks with token awareness"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk and len(chunk) > 20:  # Skip tiny chunks
                chunks.append(chunk)
        
        return chunks
    
    def document_exists(self, doc_id):
        """Check if document is already loaded"""
        try:
            results = self.collection.get(
                where={"source": doc_id}
            )
            return len(results['ids']) > 0
        except:
            return False
        
    def load_file(self, file_path, doc_id=None, use_ocr=False):
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
            self.log(f"✗ Unsupported file type: {ext}", "RAG")
    
    def load_pdf(self, pdf_path, chunk_size=200, force_reload=False, use_ocr=False):
        """Load and index a PDF file with optional OCR"""
        doc_id = os.path.basename(pdf_path)
        
        # Check if already loaded
        if not force_reload and self.document_exists(doc_id):
            self.log(f"⚠ Document '{doc_id}' already loaded. Use force_reload=True to reload.", "RAG")
            return
        
        self.log(f"Loading PDF: {pdf_path}", "RAG")
        
        try:
            # Extract text (with OCR fallback)
            text = self._extract_pdf_text(pdf_path, use_ocr=use_ocr)
            
            if not text or len(text.strip()) < 10:
                self.log(f"⚠ No text extracted from PDF even with OCR.", "RAG")
                return
            
            # Split into chunks
            chunks = self._chunk_text(text, chunk_size)
            self.log(f"Created {len(chunks)} chunks", "RAG")
            
            # Generate embeddings and store
            for i, chunk in enumerate(chunks):
                try:
                    # Try to embed the chunk
                    res = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
                    
                    self.collection.add(
                        documents=[chunk],
                        embeddings=[res['embedding']],
                        metadatas=[{"source": doc_id, "chunk_index": i, "type": "pdf"}],
                        ids=[f"{doc_id}_chunk_{i}"]
                    )
                    
                    if (i + 1) % 10 == 0:
                        self.log(f"Processed {i + 1}/{len(chunks)} chunks", "RAG")
                        
                except Exception as e:
                    if "exceeds the context length" in str(e):
                        self.log(f"⚠ Chunk {i} too large, splitting further...", "RAG")
                        # Split the chunk in half and try again
                        words = chunk.split()
                        mid = len(words) // 2
                        sub_chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
                        
                        for j, sub_chunk in enumerate(sub_chunks):
                            try:
                                res = ollama.embeddings(model=EMBED_MODEL, prompt=sub_chunk)
                                self.collection.add(
                                    documents=[sub_chunk],
                                    embeddings=[res['embedding']],
                                    metadatas=[{"source": doc_id, "chunk_index": f"{i}_{j}", "type": "pdf"}],
                                    ids=[f"{doc_id}_chunk_{i}_{j}"]
                                )
                            except Exception as sub_e:
                                self.log(f"✗ Failed to embed sub-chunk: {sub_e}", "RAG")
                    else:
                        self.log(f"✗ Error embedding chunk {i}: {e}", "RAG")
            
            self.log(f"✓ PDF loaded successfully!", "RAG")
        except Exception as e:
            self.log(f"✗ Error loading PDF: {e}", "RAG")
    
    def load_text(self, text, doc_id="text_doc", chunk_size=500):
        """Load raw text (for non-PDF documents)"""
        if not text or len(text.strip()) < 10:
            self.log(f"⚠ Text is too short or empty", "RAG")
            return
        
        chunks = self._chunk_text(text, chunk_size)
        
        for i, chunk in enumerate(chunks):
            res = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
            
            self.collection.add(
                documents=[chunk],
                embeddings=[res['embedding']],
                metadatas=[{"source": doc_id, "chunk_index": i, "type": "text"}],
                ids=[f"{doc_id}_chunk_{i}"]
            )
        
        self.log(f"✓ Text loaded: {len(chunks)} chunks", "RAG")

    def load_image(self, image_path, doc_id=None):
        """Extract text from image using OCR"""
        try:
            if doc_id is None:
                doc_id = os.path.basename(image_path)
            
            # Check if already loaded
            if self.document_exists(doc_id):
                self.log(f"⚠ Document '{doc_id}' already loaded.", "RAG")
                return
            
            self.log(f"Loading image: {image_path}", "RAG")
            
            # Extract text using OCR
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            
            if not text or len(text.strip()) < 10:
                self.log(f"⚠ No text extracted from image", "RAG")
                return
            
            # Load the extracted text
            self.load_text(text, doc_id=doc_id)
            self.log(f"✓ Image loaded successfully!", "RAG")
            
        except Exception as e:
            self.log(f"✗ Error loading image: {e}", "RAG")

    def load_docx(self, docx_path, doc_id=None):
        """Load Word document"""
        try:
            if doc_id is None:
                doc_id = os.path.basename(docx_path)
            
            if self.document_exists(doc_id):
                self.log(f"⚠ Document '{doc_id}' already loaded.", "RAG")
                return
            
            self.log(f"Loading Word document: {docx_path}", "RAG")
            
            # Extract text
            doc = Document(docx_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            
            if not text or len(text.strip()) < 10:
                self.log(f"⚠ No text extracted from Word document", "RAG")
                return
            
            # Load the text
            self.load_text(text, doc_id=doc_id)
            self.log(f"✓ Word document loaded successfully!", "RAG")
            
        except Exception as e:
            self.log(f"✗ Error loading Word document: {e}", "RAG")    

    def load_excel(self, excel_path, doc_id=None):
        """Load Excel file"""
        try:
            if doc_id is None:
                doc_id = os.path.basename(excel_path)
            
            if self.document_exists(doc_id):
                self.log(f"⚠ Document '{doc_id}' already loaded.", "RAG")
                return
            
            self.log(f"Loading Excel file: {excel_path}", "RAG")
            
            # Read all sheets
            excel_file = pd.ExcelFile(excel_path)
            all_text = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                all_text.append(f"Sheet: {sheet_name}\n{df.to_string()}")
            
            text = '\n\n'.join(all_text)
            
            if not text or len(text.strip()) < 10:
                self.log(f"⚠ No data extracted from Excel file", "RAG")
                return
            
            # Load the text
            self.load_text(text, doc_id=doc_id)
            self.log(f"✓ Excel file loaded successfully!", "RAG")
            
        except Exception as e:
            self.log(f"✗ Error loading Excel file: {e}", "RAG")

    def load_csv(self, csv_path, doc_id=None):
        """Load CSV file"""
        try:
            if doc_id is None:
                doc_id = os.path.basename(csv_path)
            
            if self.document_exists(doc_id):
                self.log(f"⚠ Document '{doc_id}' already loaded.", "RAG")
                return
            
            self.log(f"Loading CSV file: {csv_path}", "RAG")
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Convert to string format
            text = f"CSV File: {doc_id}\n\n{df.to_string()}"
            
            if not text or len(text.strip()) < 10:
                self.log(f"⚠ No data extracted from CSV file", "RAG")
                return
            
            # Load the text
            self.load_text(text, doc_id=doc_id)
            self.log(f"✓ CSV file loaded successfully!", "RAG")
            
        except Exception as e:
            self.log(f"✗ Error loading CSV file: {e}", "RAG")

    def retrieve(self, query, top_k=3):
        """Retrieve most relevant chunks for a query"""
        # Get query embedding
        q_res = ollama.embeddings(model=EMBED_MODEL, prompt=query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[q_res['embedding']],
            n_results=top_k
        )
        
        # Return documents with metadata
        if results['documents']:
            return {
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0] if results['metadatas'] else []
            }
        return {'documents': [], 'metadatas': []}
    
    def get_context(self, query, top_k=3):
        """Get formatted context string for a query"""
        results = self.retrieve(query, top_k)
        chunks = results['documents']
        return "\n\n".join(chunks)
    
    def list_documents(self):
        """List all loaded documents"""
        try:
            all_data = self.collection.get()
            sources = set()
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    if 'source' in metadata:
                        sources.add(metadata['source'])
            return list(sources)
        except:
            return []
    
    def delete_document(self, doc_id):
        """Delete a specific document"""
        try:
            self.collection.delete(
                where={"source": doc_id}
            )
            self.log(f"✓ Deleted document: {doc_id}", "RAG")
        except Exception as e:
            self.log(f"✗ Error deleting document: {e}", "RAG")
    
    def clear_collection(self):
        """Clear all documents from collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)
        self.log("✓ Collection cleared", "RAG")
    
    def get_stats(self):
        """Get collection statistics"""
        count = self.collection.count()
        docs = self.list_documents()
        return {
            'total_chunks': count,
            'documents': docs,
            'num_documents': len(docs)
        }
    
# Test code 
if __name__ == "__main__":
    rag = RAGSystem()
    
    # Load PDF with automatic OCR fallback
    rag.load_file("AI_Model/tools/sample.pdf")
    
    # Or force OCR even if text extraction works
    # rag.load_pdf("AI_Model/tools/sample.pdf", use_ocr=True, force_reload=True)
    
    context = rag.get_context("What is the main topic?", top_k=2)
    print("Retrieved Context:\n", context)