import os
import re
import json
import pickle
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from unstructured.partition.pdf import partition_pdf
from PIL import Image
import pytesseract

# Force offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TIKTOKEN_CACHE_DIR'] = os.path.expanduser('~/.cache/tiktoken')

# Handle truncated images
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class ChunkConfig:
    max_chunk_size: int = 512
    chunk_overlap: int = 50
    table_max_size: int = 2048
    image_context_window: int = 5

class LLMFriendlyTableFormatter:
    """Convert tables to natural language"""
    
    @staticmethod
    def format_table_for_llm(table_text: str, context_info: Dict) -> str:
        """Convert table into natural, descriptive text."""
        lines = table_text.strip().split('\n')
        if not lines:
            return "Empty table"
        
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        data_rows = []
        for line in lines[1:]:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells and not all(c in ['-', ''] for c in cells):
                data_rows.append(cells)
        
        parts = []
        
        if context_info.get('title'):
            parts.append(f"# {context_info['title']}\n")
        
        if context_info.get('description'):
            parts.append(f"{context_info['description']}\n")
        
        parts.append(f"This table contains {len(data_rows)} entries with {len(headers)} columns: {', '.join(headers)}.\n")
        
        parts.append("\nDetailed Data:")
        for i, row in enumerate(data_rows, 1):
            row_desc = []
            for header, value in zip(headers, row):
                if value:
                    row_desc.append(f"{header}: {value}")
            if row_desc:
                parts.append(f"{i}. {', '.join(row_desc)}")
        
        parts.append("\n[Original Table Format]")
        parts.append(" | ".join(headers))
        parts.append("-" * (len(headers) * 15))
        for row in data_rows:
            parts.append(" | ".join(row))
        
        return "\n".join(parts)


class SimpleTokenCounter:
    """Fallback token counter when tiktoken is not available."""
    
    def count_tokens(self, text: str) -> int:
        # Simple approximation: ~4 chars per token
        return len(text) // 4


class DynamicTextChunker:
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        
        # Try to use tiktoken, fallback to simple counter
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
            print("✅ Using tiktoken for accurate token counting")
        except Exception as e:
            print(f"⚠️  tiktoken not available, using fallback counter")
            self.encoder = SimpleTokenCounter()
            self.use_tiktoken = False
    
    def count_tokens(self, text: str) -> int:
        if self.use_tiktoken:
            return len(self.encoder.encode(text))
        else:
            return self.encoder.count_tokens(text)
    
    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text_semantic(self, text: str, metadata: Dict = None) -> List[Dict]:
        chunks = []
        sentences = self.split_sentences(text)
        if not sentences:
            return []
        
        current_chunk = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = self.count_tokens(sent)
            
            if current_tokens + sent_tokens > self.config.max_chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'token_count': self.count_tokens(chunk_text),
                        'metadata': {**metadata, 'chunk_index': len(chunks)} if metadata else {'chunk_index': len(chunks)}
                    })
                
                overlap_sentences = []
                overlap_tokens = 0
                for prev_sent in reversed(current_chunk[-3:]):
                    prev_tokens = self.count_tokens(prev_sent)
                    if overlap_tokens + prev_tokens <= self.config.chunk_overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_tokens += prev_tokens
                
                current_chunk = overlap_sentences + [sent]
                current_tokens = overlap_tokens + sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'token_count': self.count_tokens(chunk_text),
                'metadata': {**metadata, 'chunk_index': len(chunks)} if metadata else {'chunk_index': len(chunks)}
            })
        
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks


def extract_figure_table_number(text: str) -> Tuple[str, str]:
    """Extract figure/table number and title."""
    patterns = [
        r'(Table|Figure|Chart|Diagram)\s+(\d+(?:\.\d+)?)[:\.\s]+(.*)',
        r'(Table|Figure|Chart|Diagram)\s+(\d+(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            element_type = match.group(1).capitalize()
            number = match.group(2)
            title = match.group(3) if len(match.groups()) > 2 else ""
            full_title = f"{element_type} {number}"
            if title:
                full_title += f": {title.strip()}"
            return (number, full_title)
    
    return (None, None)


def extract_text_around_element(elements: List, element_index: int, 
                                window_size: int = 10) -> Tuple[str, str]:
    """Extract text before and after an element."""
    text_before = []
    text_after = []
    
    start_idx = max(0, element_index - window_size)
    for i in range(start_idx, element_index):
        elem = elements[i]
        elem_type = type(elem).__name__
        if elem_type in ["NarrativeText", "Text", "Title", "ListItem"]:
            text = str(elem).strip()
            if len(text) >= 20:
                text_before.append(text)
    
    end_idx = min(len(elements), element_index + window_size + 1)
    for i in range(element_index + 1, end_idx):
        elem = elements[i]
        elem_type = type(elem).__name__
        if elem_type in ["NarrativeText", "Text", "Title", "ListItem"]:
            text = str(elem).strip()
            if len(text) >= 20:
                text_after.append(text)
    
    return (' '.join(text_before[-5:]), ' '.join(text_after[:5]))


def extract_text_from_image_ocr(image_path: str) -> str:
    """Extract text from image using OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        if text and len(text.strip()) > 10:
            return f"\n**OCR Extracted Text from Image:**\n{text.strip()}"
        return ""
    except Exception as e:
        print(f"    ⚠️  OCR failed: {e}")
        return ""


def find_table_context(elements: List, table_index: int) -> Dict[str, str]:
    """Find context for a table with proper number extraction."""
    context_info = {
        'title': '',
        'description': '',
        'number': None,
        'type': 'table'
    }
    
    start_idx = max(0, table_index - 10)
    context_paras = []
    
    for i in range(start_idx, table_index):
        elem = elements[i]
        elem_type = type(elem).__name__
        
        if elem_type in ["NarrativeText", "Text", "Title"]:
            para_text = str(elem).strip()
            
            if not para_text or len(para_text) < 3:
                continue
            
            number, full_title = extract_figure_table_number(para_text)
            
            if number and not context_info['title']:
                context_info['title'] = full_title
                context_info['number'] = number
                print(f"      ✓ Found caption: {full_title}")
            elif len(para_text) < 100 and para_text.isupper() and not context_info['title']:
                context_info['title'] = para_text
            else:
                context_paras.append(para_text)
    
    if context_paras:
        context_info['description'] = ' '.join(context_paras[-5:])
    
    return context_info


def find_image_context(elements: List, image_index: int) -> Dict[str, str]:
    """Find context for an image with proper figure number extraction."""
    context_info = {
        'title': '',
        'caption': '',
        'description': '',
        'number': None,
        'type': 'figure'
    }
    
    search_range = list(range(max(0, image_index - 10), image_index))
    search_range += list(range(image_index + 1, min(len(elements), image_index + 10)))
    
    context_paras = []
    
    for i in search_range:
        elem = elements[i]
        elem_type = type(elem).__name__
        
        if elem_type in ["NarrativeText", "Text", "Title", "FigureCaption"]:
            para_text = str(elem).strip()
            
            if not para_text or len(para_text) < 3:
                continue
            
            number, full_title = extract_figure_table_number(para_text)
            
            if number and not context_info['title']:
                context_info['title'] = full_title
                context_info['caption'] = para_text
                context_info['number'] = number
                print(f"      ✓ Found caption: {full_title}")
            else:
                context_paras.append(para_text)
    
    if context_paras:
        context_info['description'] = ' '.join(context_paras[-5:])
    
    return context_info


def find_image_path(output_dir: str, element_index: int) -> str:
    """Find the actual image file path."""
    import glob
    
    patterns = [
        f"figure-{element_index}*.jpg",
        f"figure-{element_index}*.png",
        f"figure-{element_index}*.jpeg",
        f"figure-{element_index}-*.jpg",
        f"figure-{element_index}-*.png",
        f"*-{element_index}.jpg",
        f"*-{element_index}.png",
    ]
    
    for pattern in patterns:
        full_pattern = os.path.join(output_dir, pattern)
        matches = glob.glob(full_pattern)
        if matches:
            return matches[0]
    
    try:
        all_files = sorted([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        if element_index < len(all_files):
            return os.path.join(output_dir, all_files[element_index])
    except:
        pass
    
    return None


def check_models_cached() -> bool:
    """Check if YOLO models are cached."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    if not os.path.exists(cache_dir):
        return False
    
    try:
        for item in os.listdir(cache_dir):
            if 'yolo' in item.lower():
                return True
    except:
        pass
    return False


def extract_and_process_pdf(pdf_path: str, output_dir: str = "extracted_data") -> List[Dict]:
    """
    Extract PDF with proper figure/table reference tracking (100% OFFLINE).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📄 PROCESSING PDF (100% OFFLINE)")
    print(f"{'='*70}")
    print(f"File: {pdf_path}")
    
    # Check if models are cached
    models_cached = check_models_cached()
    if models_cached:
        print("✅ YOLO models found in cache - using hi_res")
        strategy = "hi_res"
    else:
        print("⚠️  YOLO models not cached - using fast (offline)")
        strategy = "fast"
    
    # Extract
    print(f"\n📄 Extracting content with {strategy} strategy...")
    try:
        raw_elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=output_dir,
        )
        print(f"✅ Extracted {len(raw_elements)} elements")
    except Exception as e:
        if "truncated" in str(e).lower() or "huggingface" in str(e).lower():
            print(f"⚠️  {strategy} failed: {str(e)[:100]}")
            print("📄 Retrying with fast strategy...")
            raw_elements = partition_pdf(
                filename=pdf_path,
                strategy="fast",
                extract_images_in_pdf=True,
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=False,
                extract_image_block_output_dir=output_dir,
            )
            print(f"✅ Extracted {len(raw_elements)} elements with fast strategy")
        else:
            raise
    
    # List extracted images
    print(f"\n🖼️  Checking extracted images...")
    try:
        image_files = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"   Found {len(image_files)} image files")
        for img_file in image_files[:3]:
            print(f"     - {img_file}")
        if len(image_files) > 3:
            print(f"     ... and {len(image_files) - 3} more")
    except:
        print("   No images found")
    
    # Initialize
    config = ChunkConfig()
    chunker = DynamicTextChunker(config)
    formatter = LLMFriendlyTableFormatter()
    all_chunks = []
    pdf_name = Path(pdf_path).name
    
    tables_with_context = []
    images_with_context = []
    texts = []
    
    # Process elements
    image_counter = 0
    table_counter = 0
    
    print("\n🔍 Processing elements...")
    
    for i, element in enumerate(raw_elements):
        element_type = type(element).__name__
        element_text = str(element)
        
        if element_type == "Table":
            print(f"\n  📊 Processing Table #{table_counter + 1}:")
            context_info = find_table_context(raw_elements, i)
            llm_friendly_table = formatter.format_table_for_llm(element_text, context_info)
            
            text_before, text_after = extract_text_around_element(raw_elements, i, window_size=10)
            
            table_with_context = []
            if text_before:
                table_with_context.append(f"**Context Before Table (5 sentences):**\n{text_before}\n")
            table_with_context.append(llm_friendly_table)
            if text_after:
                table_with_context.append(f"\n**Context After Table (5 sentences):**\n{text_after}")
            
            tables_with_context.append({
                'content': "\n".join(table_with_context),
                'title': context_info.get('title', f'Table {table_counter + 1}'),
                'description': context_info.get('description', ''),
                'number': context_info.get('number'),
                'index': table_counter,
                'text_before': text_before,
                'text_after': text_after
            })
            
            table_counter += 1
        
        elif element_type == "Image":
            print(f"\n  🖼️  Processing Image #{image_counter + 1}:")
            img_path = find_image_path(output_dir, image_counter)
            
            if img_path:
                print(f"      ✓ File: {os.path.basename(img_path)}")
            
            context_info = find_image_context(raw_elements, i)
            text_before, text_after = extract_text_around_element(raw_elements, i, window_size=10)
            
            ocr_text = ""
            if img_path and os.path.exists(img_path):
                ocr_text = extract_text_from_image_ocr(img_path)
            
            image_context_parts = []
            
            if context_info.get('title'):
                image_context_parts.append(f"# {context_info['title']}\n")
            
            if text_before:
                image_context_parts.append(f"**Context Before Image (5 sentences):**\n{text_before}")
            
            image_context_parts.append(f"**Image File:** {img_path or 'Not found'}")
            
            if ocr_text:
                image_context_parts.append(ocr_text)
            
            if text_after:
                image_context_parts.append(f"\n**Context After Image (5 sentences):**\n{text_after}")
            
            full_context = "\n\n".join(image_context_parts)
            
            images_with_context.append({
                'path': img_path,
                'context': full_context,
                'title': context_info.get('title', f'Figure {image_counter + 1}'),
                'caption': context_info.get('caption', ''),
                'number': context_info.get('number'),
                'text_before': text_before,
                'text_after': text_after,
                'ocr_text': ocr_text,
                'index': image_counter
            })
            
            image_counter += 1
        
        elif element_type in ["NarrativeText", "Text", "Title", "ListItem"]:
            if len(element_text.strip()) >= 50:
                texts.append(element_text)
    
    print(f"\n✅ Found: {len(texts)} texts, {len(tables_with_context)} tables, {len(images_with_context)} images")
    
    # Process Tables
    print("\n💾 Creating chunks...")
    for i, table_data in enumerate(tables_with_context):
        token_count = chunker.count_tokens(table_data['content'])
        
        metadata = {
            'type': 'table',
            'source': pdf_name,
            'original_index': table_data['index'],
            'chunk_index': 0,
            'total_chunks': 1,
            'modality': 'table',
            'has_image_file': False,
            'table_title': table_data['title'],
            'table_number': table_data['number'],
            'table_description': table_data['description'],
            'llm_friendly': True,
            'has_context': True,
            'context_sentences': 5,
            'reference_names': [
                table_data['title'],
                f"table {table_data['number']}" if table_data['number'] else None,
                f"Table {table_data['number']}" if table_data['number'] else None,
            ]
        }
        
        all_chunks.append({
            'id': f'{pdf_name}_table_{i}',
            'content': table_data['content'],
            'metadata': metadata,
            'token_count': token_count
        })
        
        print(f"  ✓ Table {i+1}: {table_data['title'][:60]}")
    
    # Process Text
    for i, text in enumerate(texts):
        if len(text.strip()) < 50:
            continue
        metadata = {
            'type': 'narrative_text',
            'source': pdf_name,
            'original_index': i,
            'modality': 'text',
            'has_image_file': False
        }
        text_chunks = chunker.chunk_text_semantic(text, metadata)
        for chunk in text_chunks:
            all_chunks.append({
                'id': f'{pdf_name}_text_{i}_{chunk["metadata"]["chunk_index"]}',
                'content': chunk['text'],
                'metadata': chunk['metadata'],
                'token_count': chunk['token_count']
            })
    
    # Process Images
    for i, img_data in enumerate(images_with_context):
        token_count = chunker.count_tokens(img_data['context'])
        img_path = img_data['path']
        has_file = bool(img_path and os.path.exists(img_path))
        
        metadata = {
            'type': 'image',
            'source': pdf_name,
            'original_index': i,
            'image_path': img_path,
            'chunk_index': 0,
            'total_chunks': 1,
            'modality': 'image',
            'has_image_file': has_file,
            'has_context': bool(img_data['text_before'] or img_data['text_after']),
            'has_ocr': bool(img_data['ocr_text']),
            'use_image_directly': has_file,
            'context_sentences': 5,
            'figure_title': img_data['title'],
            'figure_number': img_data['number'],
            'figure_caption': img_data['caption'],
            'reference_names': [
                img_data['title'],
                f"figure {img_data['number']}" if img_data['number'] else None,
                f"Figure {img_data['number']}" if img_data['number'] else None,
                f"fig {img_data['number']}" if img_data['number'] else None,
            ]
        }
        
        all_chunks.append({
            'id': f'{pdf_name}_image_{i}',
            'content': img_data['context'],
            'metadata': metadata,
            'token_count': token_count
        })
        
        print(f"  ✓ Figure {i+1}: {img_data['title'][:60]}")
    
    # Save
    print(f"\n💾 Saving {len(all_chunks)} chunks...")
    with open('processed_chunks.pkl', 'wb') as f:
        pickle.dump(all_chunks, f)
    with open('processed_chunks.json', 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    print("✅ Saved to processed_chunks.pkl")
    
    # Print summary
    print(f"\n📋 SUMMARY:")
    print(f"   Tables: {len(tables_with_context)}")
    for t in tables_with_context[:3]:
        print(f"      - {t['title']}")
    print(f"   Figures: {len(images_with_context)}")
    for f in images_with_context[:3]:
        print(f"      - {f['title']}")
    
    print(f"{'='*70}\n")
    
    return all_chunks


if __name__ == "__main__":
    pdf_path = "SIH/dataset/recipe_book.pdf"
    
    try:
        import pytesseract
        print("✅ OCR available (pytesseract)")
    except ImportError:
        print("⚠️  OCR not available")
    
    print("✅ 100% OFFLINE - no internet required")
    print("✅ Context: 5 sentences before/after")
    print("✅ Feature: Figure/Table number tracking\n")
    
    chunks = extract_and_process_pdf(pdf_path)
    print(f"✅ Ready for vectorstore.py!")