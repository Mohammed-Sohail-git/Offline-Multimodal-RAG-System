# master_pipeline_parallel.py - PARALLEL Multimodal Document Processing Pipeline

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Import all handlers
from pdf_handler import extract_and_process_pdf
from doc_handler import DOCXProcessor
from image_handler import ImageProcessor, ImageProcessingConfig
from audio_handler import EnhancedAudioHandler, AudioConfig
from vectorstore import UnifiedEmbeddingGenerator, create_and_save_faiss_index


# ============================================================================
# PARALLEL PROCESSING WORKER FUNCTIONS
# ============================================================================

def process_single_pdf(args):
    """Worker function for processing single PDF."""
    pdf_path, output_dir = args
    try:
        chunks = extract_and_process_pdf(pdf_path, output_dir=output_dir)
        return {
            'success': True,
            'file': os.path.basename(pdf_path),
            'chunks': chunks,
            'count': len(chunks)
        }
    except Exception as e:
        return {
            'success': False,
            'file': os.path.basename(pdf_path),
            'error': str(e),
            'chunks': []
        }


def process_single_docx(args):
    """Worker function for processing single DOCX."""
    docx_path, output_dir = args
    try:
        processor = DOCXProcessor()
        chunks = processor.process(docx_path, output_dir=output_dir)
        return {
            'success': True,
            'file': os.path.basename(docx_path),
            'chunks': chunks,
            'count': len(chunks)
        }
    except Exception as e:
        return {
            'success': False,
            'file': os.path.basename(docx_path),
            'error': str(e),
            'chunks': []
        }


def process_single_image(args):
    """Worker function for processing single image."""
    image_path, clip_model_path, session_id = args
    try:
        processor = ImageProcessor(
            clip_model_path=clip_model_path,
            config=ImageProcessingConfig()
        )
        chunk = processor.process_image(
            image_path=image_path,
            session_id=session_id
        )
        return {
            'success': True,
            'file': os.path.basename(image_path),
            'chunks': [chunk],
            'count': 1
        }
    except Exception as e:
        return {
            'success': False,
            'file': os.path.basename(image_path),
            'error': str(e),
            'chunks': []
        }


def process_single_audio(args):
    """Worker function for processing single audio."""
    audio_path, session_id = args
    try:
        config = AudioConfig(num_speakers=None)
        handler = EnhancedAudioHandler(config)
        chunks = handler.process_audio(audio_path, session_id=session_id)
        return {
            'success': True,
            'file': os.path.basename(audio_path),
            'chunks': chunks,
            'count': len(chunks)
        }
    except Exception as e:
        return {
            'success': False,
            'file': os.path.basename(audio_path),
            'error': str(e),
            'chunks': []
        }


# ============================================================================
# PARALLEL MASTER PIPELINE
# ============================================================================

class ParallelMasterPipeline:
    """
    PARALLEL Multimodal Document Processing Pipeline.
    
    Features:
    - Parallel processing of PDFs, DOCX, Images, Audio
    - Batch processing within each type
    - Progress tracking with tqdm
    - Error handling and reporting
    - 5-10x speed improvement
    """
    
    def __init__(
        self,
        clip_model_path: str = "SIH/clip-model",
        output_dir: str = "processed_output",
        max_workers: int = None
    ):
        self.clip_model_path = clip_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Auto-detect optimal worker count
        if max_workers is None:
            # Use 75% of available CPU cores
            max_workers = max(1, int(mp.cpu_count() * 0.75))
        self.max_workers = max_workers
        
        print(f"\n{'='*70}")
        print(f"🚀 PARALLEL MULTIMODAL RAG PIPELINE")
        print(f"{'='*70}")
        print(f"Output Directory: {self.output_dir}")
        print(f"CLIP Model: {clip_model_path}")
        print(f"CPU Cores: {mp.cpu_count()}")
        print(f"Workers: {self.max_workers} (parallel)")
        print(f"Mode: PARALLEL PROCESSING ⚡")
        print(f"{'='*70}\n")
    
    def find_all_files(self, input_dir: str) -> Dict[str, List[str]]:
        """Find all supported files."""
        print(f"🔍 Scanning directory: {input_dir}")
        
        files = {
            'pdfs': [],
            'docx': [],
            'images': [],
            'audio': []
        }
        
        # File extensions
        pdf_exts = {'.pdf'}
        docx_exts = {'.docx', '.doc'}
        image_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        audio_exts = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac'}
        
        # Walk directory
        for root, dirs, filenames in os.walk(input_dir):
            for filename in filenames:
                ext = Path(filename).suffix.lower()
                full_path = os.path.join(root, filename)
                
                if ext in pdf_exts:
                    files['pdfs'].append(full_path)
                elif ext in docx_exts:
                    files['docx'].append(full_path)
                elif ext in image_exts:
                    if 'extracted_data' not in root and 'docx_extracted' not in root:
                        files['images'].append(full_path)
                elif ext in audio_exts:
                    files['audio'].append(full_path)
        
        # Print summary
        total = sum(len(v) for v in files.values())
        print(f"\n📊 Found: {total} files")
        print(f"   PDFs: {len(files['pdfs'])}")
        print(f"   DOCX: {len(files['docx'])}")
        print(f"   Images: {len(files['images'])}")
        print(f"   Audio: {len(files['audio'])}\n")
        
        return files
    
    def process_pdfs_parallel(self, pdf_paths: List[str]) -> List[Dict]:
        """Process PDFs in PARALLEL."""
        if not pdf_paths:
            return []
        
        print(f"\n{'='*70}")
        print(f"📄 PROCESSING {len(pdf_paths)} PDFs (PARALLEL)")
        print(f"{'='*70}")
        
        output_dir = str(self.output_dir / "pdf_extracted")
        
        # Prepare arguments
        args_list = [(pdf_path, output_dir) for pdf_path in pdf_paths]
        
        # Process in parallel
        all_chunks = []
        successful = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_pdf, args): args[0] 
                      for args in args_list}
            
            # Progress bar
            with tqdm(total=len(pdf_paths), desc="Processing PDFs") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result['success']:
                        all_chunks.extend(result['chunks'])
                        successful += 1
                        pbar.set_postfix({'✓': successful, '✗': failed, 'chunks': len(all_chunks)})
                    else:
                        failed += 1
                        print(f"\n   ❌ {result['file']}: {result['error']}")
                    
                    pbar.update(1)
        
        print(f"\n✅ PDFs: {successful} succeeded, {failed} failed")
        print(f"   Total chunks: {len(all_chunks)}\n")
        
        # Save
        if all_chunks:
            output_file = self.output_dir / "pdf_chunks.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(all_chunks, f)
            print(f"💾 Saved to {output_file}\n")
        
        return all_chunks
    
    def process_docx_parallel(self, docx_paths: List[str]) -> List[Dict]:
        """Process DOCX files in PARALLEL."""
        if not docx_paths:
            return []
        
        print(f"\n{'='*70}")
        print(f"📝 PROCESSING {len(docx_paths)} DOCX (PARALLEL)")
        print(f"{'='*70}")
        
        output_dir = str(self.output_dir / "docx_extracted")
        
        # Prepare arguments
        args_list = [(docx_path, output_dir) for docx_path in docx_paths]
        
        # Process in parallel
        all_chunks = []
        successful = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single_docx, args): args[0] 
                      for args in args_list}
            
            with tqdm(total=len(docx_paths), desc="Processing DOCX") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result['success']:
                        all_chunks.extend(result['chunks'])
                        successful += 1
                        pbar.set_postfix({'✓': successful, '✗': failed, 'chunks': len(all_chunks)})
                    else:
                        failed += 1
                        print(f"\n   ❌ {result['file']}: {result['error']}")
                    
                    pbar.update(1)
        
        print(f"\n✅ DOCX: {successful} succeeded, {failed} failed")
        print(f"   Total chunks: {len(all_chunks)}\n")
        
        # Save
        if all_chunks:
            output_file = self.output_dir / "docx_chunks.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(all_chunks, f)
            print(f"💾 Saved to {output_file}\n")
        
        return all_chunks
    
    def process_images_parallel(self, image_paths: List[str]) -> List[Dict]:
        """Process images in PARALLEL."""
        if not image_paths:
            return []
        
        print(f"\n{'='*70}")
        print(f"🖼️  PROCESSING {len(image_paths)} IMAGES (PARALLEL)")
        print(f"{'='*70}")
        
        # Prepare arguments
        args_list = [(img_path, self.clip_model_path, "standalone_images") 
                     for img_path in image_paths]
        
        # Process in parallel
        all_chunks = []
        successful = 0
        failed = 0
        
        # Use ThreadPoolExecutor for I/O-bound image processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single_image, args): args[0] 
                      for args in args_list}
            
            with tqdm(total=len(image_paths), desc="Processing Images") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result['success']:
                        all_chunks.extend(result['chunks'])
                        successful += 1
                        pbar.set_postfix({'✓': successful, '✗': failed, 'chunks': len(all_chunks)})
                    else:
                        failed += 1
                        print(f"\n   ❌ {result['file']}: {result['error']}")
                    
                    pbar.update(1)
        
        print(f"\n✅ Images: {successful} succeeded, {failed} failed")
        print(f"   Total chunks: {len(all_chunks)}\n")
        
        # Save
        if all_chunks:
            output_file = self.output_dir / "image_chunks.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(all_chunks, f)
            print(f"💾 Saved to {output_file}\n")
        
        return all_chunks
    
    def process_audio_parallel(self, audio_paths: List[str]) -> List[Dict]:
        """Process audio files in PARALLEL."""
        if not audio_paths:
            return []
        
        print(f"\n{'='*70}")
        print(f"🎧 PROCESSING {len(audio_paths)} AUDIO (PARALLEL)")
        print(f"{'='*70}")
        
        # Prepare arguments
        args_list = [(audio_path, "audio_files") for audio_path in audio_paths]
        
        # Process in parallel
        all_chunks = []
        successful = 0
        failed = 0
        
        # Audio is CPU-intensive, use ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single_audio, args): args[0] 
                      for args in args_list}
            
            with tqdm(total=len(audio_paths), desc="Processing Audio") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result['success']:
                        all_chunks.extend(result['chunks'])
                        successful += 1
                        pbar.set_postfix({'✓': successful, '✗': failed, 'chunks': len(all_chunks)})
                    else:
                        failed += 1
                        print(f"\n   ❌ {result['file']}: {result['error']}")
                    
                    pbar.update(1)
        
        print(f"\n✅ Audio: {successful} succeeded, {failed} failed")
        print(f"   Total chunks: {len(all_chunks)}\n")
        
        # Save
        if all_chunks:
            output_file = self.output_dir / "audio_chunks.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(all_chunks, f)
            print(f"💾 Saved to {output_file}\n")
        
        return all_chunks
    
    def process_all_types_parallel(self, files: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """
        Process ALL file types in PARALLEL simultaneously.
        
        This runs PDF, DOCX, Image, and Audio processing AT THE SAME TIME!
        Maximum speed optimization.
        """
        print(f"\n{'='*70}")
        print(f"⚡ PARALLEL PROCESSING ALL TYPES SIMULTANEOUSLY")
        print(f"{'='*70}\n")
        
        results = {}
        
        # Submit all processing tasks to run in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Create futures for each file type
            future_to_type = {
                executor.submit(self.process_pdfs_parallel, files['pdfs']): 'pdfs',
                executor.submit(self.process_docx_parallel, files['docx']): 'docx',
                executor.submit(self.process_images_parallel, files['images']): 'images',
                executor.submit(self.process_audio_parallel, files['audio']): 'audio'
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_type):
                file_type = future_to_type[future]
                try:
                    chunks = future.result()
                    results[file_type] = chunks
                    print(f"✅ {file_type.upper()} processing complete: {len(chunks)} chunks\n")
                except Exception as e:
                    print(f"❌ {file_type.upper()} processing failed: {e}\n")
                    results[file_type] = []
        
        return results
    
    def merge_all_chunks(self, results: Dict[str, List[Dict]]) -> List[Dict]:
        """Merge all chunks from different sources."""
        print(f"\n{'='*70}")
        print(f"🔗 MERGING ALL CHUNKS")
        print(f"{'='*70}\n")
        
        all_chunks = []
        for file_type, chunks in results.items():
            all_chunks.extend(chunks)
        
        print(f"📊 Merged Chunks:")
        for file_type, chunks in results.items():
            print(f"   {file_type.upper()}: {len(chunks)}")
        print(f"   TOTAL: {len(all_chunks)}\n")
        
        # Save unified chunks
        output_file = self.output_dir / "all_chunks.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(all_chunks, f)
        print(f"💾 Saved unified chunks to {output_file}")
        
        # Save summary
        summary = self._generate_summary(all_chunks, results)
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Saved summary to {summary_file}\n")
        
        return all_chunks
    
    def _generate_summary(self, chunks: List[Dict], results: Dict[str, List[Dict]]) -> Dict:
        """Generate processing summary."""
        from collections import Counter
        
        type_counts = Counter(c['metadata']['type'] for c in chunks)
        modality_counts = Counter(c['metadata']['modality'] for c in chunks)
        source_counts = Counter(c['metadata']['source'] for c in chunks)
        
        total_tokens = sum(c.get('token_count', 0) for c in chunks)
        chunks_with_images = sum(1 for c in chunks if c['metadata'].get('has_image_file', False))
        
        return {
            'timestamp': datetime.now().isoformat(),
            'processing_mode': 'parallel',
            'workers': self.max_workers,
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'by_type': dict(type_counts),
            'by_modality': dict(modality_counts),
            'by_source': dict(source_counts),
            'by_file_type': {k: len(v) for k, v in results.items()},
            'chunks_with_images': chunks_with_images,
            'cross_modal_enabled': True
        }
    
    def create_vectorstore(self, chunks: List[Dict], index_name: str = "multimodal_index") -> str:
        """Create FAISS vectorstore from all chunks."""
        print(f"\n{'='*70}")
        print(f"🔢 CREATING VECTORSTORE")
        print(f"{'='*70}\n")
        
        print("Generating CLIP embeddings...")
        generator = UnifiedEmbeddingGenerator()
        embeddings_data = generator.process_chunks(chunks)
        
        embeddings = embeddings_data['embeddings']
        metadata = embeddings_data['metadata']
        
        if len(embeddings) == 0:
            raise ValueError("No data could be extracted from the uploaded files. Check server logs for specific file processing errors.")
            
        print(f"✅ Generated {len(embeddings)} embeddings\n")
        
        index_path = str(self.output_dir / f"{index_name}.bin")
        create_and_save_faiss_index(embeddings, metadata, index_path)
        
        return index_path
    
    def run_complete_pipeline(self, input_dir: str) -> Dict:
        """Run complete PARALLEL pipeline."""
        start_time = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"🚀 STARTING PARALLEL PIPELINE")
        print(f"{'='*70}")
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input: {input_dir}")
        print(f"Workers: {self.max_workers}")
        print(f"Mode: ⚡ PARALLEL")
        print(f"{'='*70}\n")
        
        # Step 1: Find all files
        files = self.find_all_files(input_dir)
        
        # Step 2: Process ALL types in parallel
        results = self.process_all_types_parallel(files)
        
        # Step 3: Merge all chunks
        all_chunks = self.merge_all_chunks(results)
        
        # Step 4: Create vectorstore
        index_path = self.create_vectorstore(all_chunks)
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"✅ PARALLEL PIPELINE COMPLETE!")
        print(f"{'='*70}")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"Workers: {self.max_workers} parallel")
        print(f"Total Chunks: {len(all_chunks)}")
        print(f"Speed: {len(all_chunks)/duration:.1f} chunks/second")
        print(f"Vectorstore: {index_path}")
        print(f"{'='*70}\n")
        
        return {
            'chunks_file': str(self.output_dir / "all_chunks.pkl"),
            'index_path': index_path,
            'summary_file': str(self.output_dir / "processing_summary.json"),
            'total_chunks': len(all_chunks),
            'duration_seconds': duration,
            'workers': self.max_workers,
            'chunks_per_second': len(all_chunks) / duration,
            'cross_modal_enabled': True
        }


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PARALLEL Cross-Modal Multimodal RAG Pipeline'
    )
    parser.add_argument('input_dir', help='Directory containing all files')
    parser.add_argument('--output-dir', default='processed_output',
                       help='Output directory')
    parser.add_argument('--clip-model', 
                       default='SIH/clip-model',
                       help='Path to CLIP model')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: 75%% of CPU cores)')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = ParallelMasterPipeline(
        clip_model_path=args.clip_model,
        output_dir=args.output_dir,
        max_workers=args.workers
    )
    
    results = pipeline.run_complete_pipeline(args.input_dir)
    
    print("\n📋 Results:")
    print(f"   Chunks: {results['chunks_file']}")
    print(f"   Index: {results['index_path']}")
    print(f"   Total: {results['total_chunks']} chunks")
    print(f"   Time: {results['duration_seconds']:.1f}s")
    print(f"   Speed: {results['chunks_per_second']:.1f} chunks/s")
    print(f"   Workers: {results['workers']} parallel")
    print(f"   Cross-Modal: ✅\n")


if __name__ == "__main__":
    main()