import os
from huggingface_hub import snapshot_download

def download_models():
    # Make sure we are not in offline mode for downloading
    if 'HF_HUB_OFFLINE' in os.environ:
        del os.environ['HF_HUB_OFFLINE']
    
    print("Downloading CLIP model for offline use...")
    snapshot_download(
        repo_id="sentence-transformers/clip-ViT-B-32", 
        local_dir="SIH/clip-model",
        local_dir_use_symlinks=False
    )
    
    print("Downloading Reranker model for offline use...")
    snapshot_download(
        repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        local_dir="SIH/cross-encoder_ms-marco-MiniLM-L-6-v2",
        local_dir_use_symlinks=False
    )
    
    print("Downloading Faster-Whisper Small for offline audio transcription...")
    snapshot_download(
        repo_id="Systran/faster-whisper-small", 
        local_dir="SIH/faster-whisper-small",
        local_dir_use_symlinks=False
    )
    
    print("Models successfully downloaded to the 'SIH' directory!")

if __name__ == "__main__":
    download_models()
