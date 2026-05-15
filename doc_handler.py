# doc_handler.py - FIXED: Image context + OCR + Direct LLaVA support

import os
import pickle
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import tiktoken
from io import BytesIO
from PIL import Image

# Try to import pytesseract for OCR
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  OCR not available. Install: brew install tesseract && pip install pytesseract")


class LLMFriendlyTableFormatter:
    """Convert tables into natural language that LLMs understand."""
    
    @staticmethod
    def format_table_for_llm(table: Table, context_info: Dict) -> str:
        """Convert table into natural, descriptive text."""
        # Extract table data
        rows = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            rows.append(row_data)
        
        if not rows:
            return "Empty table"
        
        # Assume first row is header
        headers = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        
        # Build natural language description
        parts = []
        
        # Add context
        if context_info.get('title'):
            parts.append(f"# {context_info['title']}\n")
        
        if context_info.get('description'):
            parts.append(f"{context_info['description']}\n")
        
        # Describe table structure
        parts.append(f"This table contains {len(data_rows)} entries with {len(headers)} columns: {', '.join(headers)}.\n")
        
        # Convert each row to natural language
        parts.append("\nDetailed Data:")
        
        for i, row in enumerate(data_rows, 1):
            row_description = []
            for header, value in zip(headers, row):
                if value:
                    row_description.append(f"{header}: {value}")
            
            if row_description:
                parts.append(f"{i}. {', '.join(row_description)}")
        
        # Add summary statistics
        summary = LLMFriendlyTableFormatter._generate_summary(headers, data_rows)
        if summary:
            parts.append(f"\nSummary: {summary}")
        
        # Add original table format
        parts.append("\n[Original Table Format]")
        parts.append(" | ".join(headers))
        parts.append("-" * (len(headers) * 15))
        for row in data_rows:
            parts.append(" | ".join(row))
        
        return "\n".join(parts)
    
    @staticmethod
    def _generate_summary(headers: List[str], data_rows: List[List[str]]) -> str:
        """Generate summary statistics for numeric columns."""
        summaries = []
        
        for col_idx, header in enumerate(headers):
            numeric_values = []
            for row in data_rows:
                if col_idx < len(row):
                    value = row[col_idx]
                    clean_value = re.sub(r'[^\d.]', '', value)
                    try:
                        numeric_values.append(float(clean_value))
                    except ValueError:
                        continue
            
            if numeric_values and len(numeric_values) >= len(data_rows) * 0.5:
                max_val = max(numeric_values)
                
                max_idx = -1
                for i, row in enumerate(data_rows):
                    if col_idx < len(row):
                        clean = re.sub(r'[^\d.]', '', row[col_idx])
                        try:
                            if float(clean) == max_val:
                                max_idx = i
                                break
                        except ValueError:
                            pass
                
                if max_idx >= 0 and len(data_rows[max_idx]) > 0:
                    leader = data_rows[max_idx][0]
                    summaries.append(f"The highest {header} is {max_val} (from {leader})")
        
        return "; ".join(summaries) if summaries else ""


class DOCXProcessor:
    """Process DOCX with image context preservation + direct LLaVA support."""
    
    def __init__(self):
        """Initialize processor."""
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.formatter = LLMFriendlyTableFormatter()
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def extract_text_from_image_ocr(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        if not OCR_AVAILABLE:
            return ""
        
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            if text and len(text.strip()) > 10:
                return f"\n**OCR Extracted Text from Image:**\n{text.strip()}"
            return ""
        except Exception as e:
            print(f"    ⚠️  OCR failed: {e}")
            return ""
    
    def extract_images_from_docx(self, doc_path: str, output_dir: str) -> List[Dict]:
        """Extract embedded images from DOCX."""
        doc = Document(doc_path)
        images = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    image = Image.open(BytesIO(image_data))
                    
                    doc_name = Path(doc_path).stem
                    image_filename = f"{doc_name}_image_{len(images)}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    image.save(image_path)
                    
                    images.append({
                        'path': image_path,
                        'index': len(images)
                    })
                    print(f"  📷 Extracted image: {image_filename}")
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not extract image: {e}")
        
        return images
    
    def find_table_context(self, doc_elements: List, table_index: int, 
                          lookback: int = 5) -> Dict[str, str]:
        """Find context for a table by looking at surrounding paragraphs."""
        context_info = {
            'title': '',
            'description': ''
        }
        
        start_idx = max(0, table_index - lookback)
        context_paras = []
        
        for i in range(start_idx, table_index):
            if isinstance(doc_elements[i], Paragraph):
                para_text = doc_elements[i].text.strip()
                
                if not para_text or len(para_text) < 3:
                    continue
                
                # Check if heading
                is_heading = (
                    len(para_text) < 100 and
                    (para_text.isupper() or
                     (hasattr(doc_elements[i], 'style') and 
                      doc_elements[i].style.name.startswith('Heading')) or
                     re.match(r'^(Table|Figure|Chart)\s*\d*[:.]\s*', para_text, re.IGNORECASE))
                )
                
                if is_heading and not context_info['title']:
                    context_info['title'] = para_text
                else:
                    context_paras.append(para_text)
        
        if context_paras:
            context_info['description'] = ' '.join(context_paras[-3:])
        
        return context_info
    
    def find_image_context(self, doc_elements: List, image_index: int, 
                          window_size: int = 5) -> Tuple[str, str]:
        """
        Extract text before and after an image.
        
        Returns: (text_before, text_after)
        """
        text_before = []
        text_after = []
        
        # Text before image
        start_idx = max(0, image_index - window_size)
        for i in range(start_idx, image_index):
            if isinstance(doc_elements[i], Paragraph):
                text = doc_elements[i].text.strip()
                if len(text) >= 20:
                    text_before.append(text)
        
        # Text after image
        end_idx = min(len(doc_elements), image_index + window_size + 1)
        for i in range(image_index + 1, end_idx):
            if isinstance(doc_elements[i], Paragraph):
                text = doc_elements[i].text.strip()
                if len(text) >= 20:
                    text_after.append(text)
        
        return (' '.join(text_before[-3:]),  # Last 3 paragraphs before
                ' '.join(text_after[:3]))     # First 3 paragraphs after
    
    def extract_document_structure(self, doc: Document) -> List:
        """Extract document elements in order."""
        elements = []
        
        for element in doc.element.body:
            if element.tag.endswith('p'):
                para = Paragraph(element, doc)
                elements.append(para)
            elif element.tag.endswith('tbl'):
                table = Table(element, doc)
                elements.append(table)
        
        return elements
    
    def process(self, docx_path: str, output_dir: str = "extracted_data1") -> List[Dict]:
        """
        Process DOCX with:
        1. LLM-friendly tables with context
        2. Images with surrounding text + OCR
        3. Direct image paths for LLaVA
        
        Returns:
            List of chunk dictionaries
        """
        print(f"\n{'='*70}")
        print(f"📄 PROCESSING DOCX WITH ENHANCED EXTRACTION")
        print(f"{'='*70}")
        print(f"File: {docx_path}")
        
        doc = Document(docx_path)
        doc_name = Path(docx_path).name
        
        # Extract document structure
        print("\n📄 Extracting document structure...")
        doc_elements = self.extract_document_structure(doc)
        
        # Track elements
        paragraphs = []
        tables_with_context = []
        images_with_context = []
        image_files = []
        
        # First pass: extract images to get their indices
        print("\n🖼️  Extracting images...")
        image_files = self.extract_images_from_docx(docx_path, output_dir)
        
        # Second pass: process elements with context
        image_counter = 0
        
        for idx, element in enumerate(doc_elements):
            if isinstance(element, Paragraph):
                text = element.text.strip()
                if len(text) >= 50:
                    paragraphs.append(text)
                
                # Check if paragraph contains image reference
                # (This is a simplified check - DOCX images are embedded differently)
                
            elif isinstance(element, Table):
                # Get context for table
                context_info = self.find_table_context(doc_elements, idx, lookback=5)
                
                # Format table for LLM
                llm_friendly_content = self.formatter.format_table_for_llm(
                    element, 
                    context_info
                )
                
                tables_with_context.append({
                    'content': llm_friendly_content,
                    'title': context_info.get('title', ''),
                    'description': context_info.get('description', ''),
                    'index': len(tables_with_context)
                })
                
                print(f"  📊 Formatted table: {context_info.get('title', 'Untitled')[:50]}")
        
        # Process image context
        # Note: DOCX doesn't preserve image position in element tree well,
        # so we approximate based on document flow
        print("\n🖼️  Building image context...")
        for i, img_file in enumerate(image_files):
            # Estimate position (rough approximation)
            estimated_position = int((i / len(image_files)) * len(doc_elements))
            
            # Get surrounding text
            text_before, text_after = self.find_image_context(
                doc_elements, 
                estimated_position, 
                window_size=5
            )
            
            # Get OCR text
            ocr_text = self.extract_text_from_image_ocr(img_file['path'])
            
            # Build comprehensive context
            image_context_parts = []
            
            if text_before:
                image_context_parts.append(f"**Context Before Image:**\n{text_before}")
            
            image_context_parts.append(f"**Image File:** {img_file['path']}")
            
            if ocr_text:
                image_context_parts.append(ocr_text)
            
            if text_after:
                image_context_parts.append(f"\n**Context After Image:**\n{text_after}")
            
            full_context = "\n\n".join(image_context_parts)
            
            images_with_context.append({
                'path': img_file['path'],
                'context': full_context,
                'text_before': text_before,
                'text_after': text_after,
                'ocr_text': ocr_text,
                'index': i
            })
            
            print(f"  🖼️  Image {i+1} with context")
        
        print(f"✅ Found: {len(paragraphs)} paragraphs, {len(tables_with_context)} tables, {len(images_with_context)} images")
        
        all_chunks = []
        
        # Process paragraphs
        for i, para_text in enumerate(paragraphs):
            token_count = self.count_tokens(para_text)
            
            chunk = {
                'id': f'{doc_name}_para_{i}',
                'content': para_text,
                'metadata': {
                    'type': 'narrative_text',
                    'source': doc_name,
                    'original_index': i,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'modality': 'text',
                    'has_image_file': False
                },
                'token_count': token_count
            }
            all_chunks.append(chunk)
        
        # Process tables (LLM-friendly format)
        for i, table_data in enumerate(tables_with_context):
            token_count = self.count_tokens(table_data['content'])
            
            chunk = {
                'id': f'{doc_name}_table_{i}',
                'content': table_data['content'],
                'metadata': {
                    'type': 'table',
                    'source': doc_name,
                    'original_index': table_data['index'],
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'modality': 'table',
                    'has_image_file': False,
                    'table_title': table_data['title'],
                    'table_description': table_data['description'],
                    'has_context': bool(table_data['title'] or table_data['description']),
                    'llm_friendly': True
                },
                'token_count': token_count
            }
            all_chunks.append(chunk)
        
        # Process images with FULL context + direct path for LLaVA
        print("\n🖼️  Finalizing image chunks...")
        for i, img_data in enumerate(images_with_context):
            # Store BOTH: context text + image path
            token_count = self.count_tokens(img_data['context'])
            img_path = img_data['path']
            has_file = bool(img_path and os.path.exists(img_path))
            
            chunk = {
                'id': f'{doc_name}_image_{i}',
                'content': img_data['context'],  # Text context for embedding
                'metadata': {
                    'type': 'image',
                    'source': doc_name,
                    'original_index': i,
                    'image_path': img_path,  # CRITICAL: For LLaVA
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'modality': 'image',
                    'has_image_file': has_file,
                    'has_context': bool(img_data['text_before'] or img_data['text_after']),
                    'has_ocr': bool(img_data['ocr_text']),
                    'use_image_directly': has_file  # NEW: Flag for LLaVA
                },
                'token_count': token_count
            }
            all_chunks.append(chunk)
        
        print(f"\n💾 Created {len(all_chunks)} chunks in enhanced format")
        print(f"{'='*70}\n")
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict], output_prefix: str = "docx_chunks"):
        """Save chunks with enhanced metadata."""
        # Save as pickle
        pickle_file = f"{output_prefix}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"💾 Saved chunks to: {pickle_file}")
        
        # Save as JSON
        json_file = f"{output_prefix}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved chunks to: {json_file}")
        
        # Create summary
        summary_file = f"{output_prefix}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"DOCX Processing Summary (Enhanced)\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total chunks: {len(chunks)}\n\n")
            
            # Statistics
            chunk_types = {}
            images_with_context = 0
            tables_with_context = 0
            
            for chunk in chunks:
                chunk_type = chunk['metadata']['type']
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                if chunk_type == 'table' and chunk['metadata'].get('has_context'):
                    tables_with_context += 1
                if chunk_type == 'image' and chunk['metadata'].get('has_context'):
                    images_with_context += 1
            
            f.write("Chunk Distribution:\n")
            for chunk_type, count in chunk_types.items():
                f.write(f"  {chunk_type}: {count}\n")
            
            f.write(f"\nEnhancements:\n")
            f.write(f"  Tables with context: {tables_with_context}/{chunk_types.get('table', 0)}\n")
            f.write(f"  Images with context: {images_with_context}/{chunk_types.get('image', 0)}\n")
            f.write(f"  OCR enabled: {OCR_AVAILABLE}\n")
            
            f.write(f"\n{'='*60}\n")
        
        print(f"💾 Saved summary to: {summary_file}")
        
        return pickle_file, json_file, summary_file


def test_docx_processor():
    """Test the enhanced DOCX processor."""
    doc_path = "dataset/sample3.docx" 
    pdf_path = "dataset/recipe_book.pdf"
    
    if not os.path.exists(doc_path):
        print(f"❌ Error: File not found at {doc_path}")
        print("Please update the path to your DOCX file")
        return None
    
    # Check OCR
    if OCR_AVAILABLE:
        print("✅ OCR available (pytesseract)")
    else:
        print("⚠️  OCR not available - text in images won't be extracted")
    
    # Initialize processor
    processor = DOCXProcessor()
    
    # Process document
    chunks = processor.process(doc_path, output_dir="extracted_data")
    
    # Save chunks
    print(f"\n{'='*70}")
    print("💾 SAVING CHUNKS TO DISK")
    print(f"{'='*70}\n")
    
    doc_name = Path(doc_path).stem
    pickle_file, json_file, summary_file = processor.save_chunks(
        chunks, 
        output_prefix=f"{doc_name}_chunks_enhanced"
    )
    
    # Display results
    print(f"\n{'='*70}")
    print(f"✅ Processing Complete!")
    print(f"{'='*70}")
    print(f"Total chunks: {len(chunks)}")
    
    # Statistics
    chunk_types = {}
    for chunk in chunks:
        chunk_type = chunk['metadata']['type']
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    print("\nChunk Distribution:")
    for chunk_type, count in chunk_types.items():
        print(f"  {chunk_type}: {count}")
    
    # Display saved files
    print("\n📁 Saved Files:")
    print(f"  1. {pickle_file} (for vectorstore)")
    print(f"  2. {json_file} (for inspection)")
    print(f"  3. {summary_file} (report)")
    print(f"  4. extracted_data/ (images)")
    
    # Show sample
    print("\n🖼️  Sample Image Chunk:")
    image_chunks = [c for c in chunks if c['metadata']['type'] == 'image']
    if image_chunks:
        example = image_chunks[0]
        print(f"\nImage ID: {example['id']}")
        print(f"Has context: {example['metadata'].get('has_context', False)}")
        print(f"Has OCR: {example['metadata'].get('has_ocr', False)}")
        print(f"Use directly: {example['metadata'].get('use_image_directly', False)}")
        print(f"Tokens: {example['token_count']}\n")
        print("Content preview:")
        print("-"*70)
        print(example['content'][:400])
        print("...")
        print("-"*70)
    
    print(f"\n{'='*70}")
    print("✅ Tables have context!")
    print("✅ Images have surrounding text + OCR!")
    print("✅ Direct image paths for LLaVA!")
    print("✅ Ready for vectorstore.py!")
    print(f"{'='*70}\n")
    
    return chunks


def load_chunks(pickle_file: str = "docx_chunks_enhanced.pkl") -> List[Dict]:
    """Load previously saved chunks."""
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Chunk file not found: {pickle_file}")
    
    with open(pickle_file, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"✅ Loaded {len(chunks)} chunks from {pickle_file}")
    return chunks


if __name__ == "__main__":
    test_docx_processor()