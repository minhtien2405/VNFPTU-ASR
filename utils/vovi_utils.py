import os
import logging
from typing import Dict, Optional
from transformers import (
	TrainerCallback,
)
import wandb
import re
import requests
import librosa
import logging

class WandbCallback(TrainerCallback):
	def on_evaluate(self, args, state, control, metrics, **kwargs):
		if "eval_wer" in metrics:
			wandb.log({"eval_wer": metrics["eval_wer"], "step": state.global_step})

def download_audio_from_s3(url: str, cache_dir: str = "./audio_cache") -> Optional[str]:
    """Download audio file from S3 URL and return local path, preserving original extension."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use the filename directly from the URL, which is correct (e.g., ends with .aac)
        if '/' in url:
            filename = url.split('/')[-1]
        else:
            # Fallback for unusual URLs
            filename = f"{hash(url)}.audio" 

        local_path = os.path.join(cache_dir, filename)
        
        # Check if file already exists and is valid
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
            
        # Download the file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        # Write file with temporary name first, then rename (atomic operation)
        temp_path = local_path + '.tmp'
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        
        os.rename(temp_path, local_path)
        
        # Verify file was downloaded correctly
        if os.path.getsize(local_path) == 0:
            os.remove(local_path)
            raise ValueError("Downloaded file is empty")
            
        return local_path
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error downloading audio from {url}: {str(e)}")
        # Clean up temp file if it exists
        temp_path = os.path.join(cache_dir, filename + '.tmp') if 'filename' in locals() else None
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def load_audio_from_path(audio_path: str, target_sr: int = 16000) -> Optional[Dict]:
	"""Load audio from local path and return audio dict."""
	try:
		if not os.path.exists(audio_path):
			return None
			
		# Load audio using librosa
		audio_array, sr = librosa.load(audio_path, sr=target_sr)
		
		return {
			"array": audio_array,
			"sampling_rate": target_sr,
			"path": audio_path
		}
	except Exception as e:
		logging.getLogger(__name__).error(f"Error loading audio from {audio_path}: {str(e)}")
		return None

def validate_audio(sample, logger: logging.Logger) -> bool:
	"""Validate audio sample for non-empty and valid format."""
	try:
		audio_data = sample.get("audio")
		
		if audio_data is None:
			logger.warning(f"No audio data found in sample")
			return False
			
		if audio_data["array"] is None or len(audio_data["array"]) == 0:
			logger.warning(f"Invalid audio array in sample")
			return False
			
		# Additional validation: check audio duration (optional)
		duration = len(audio_data["array"]) / audio_data["sampling_rate"]
		if duration < 0.1 or duration > 30:  # 0.1s to 30s reasonable range
			logger.warning(f"Audio duration {duration:.2f}s is outside reasonable range")
			return False
			
		return True
	except Exception as e:
		logger.error(f"Error validating audio sample: {str(e)}")
		return False
	
def validate_text(sample, logger: logging.Logger) -> bool:
	try:
		text = sample.get("text", "")
		
		if not text or len(text) == 0:
			logger.warning(f"Empty text found in sample")
			return False
			
		if len(text) > 1000:
			logger.warning(f"Text too long ({len(text)} chars): {text[:100]}...")
			return False
			
		# Additional validation: check for reasonable Vietnamese text
		if len(text.strip()) < 2:
			logger.warning(f"Text too short: '{text}'")
			return False
			
		return True
	except Exception as e:
		logger.error(f"Error validating text sample: {str(e)}")
		return False

def normalize_text(text: str) -> str:
	"""Clean and normalize text labels."""
	try:
		text = text.lower() 
		text = re.sub(r'[^\w\s]', '', text)
		text = re.sub(r'\s+', ' ', text).strip()
		return text
	except Exception as e:
		logging.getLogger(__name__).error(f"Error normalizing text: {str(e)}")
		return text

def process_audio_sample(sample, logger: logging.Logger) -> Dict:
	"""Process a single audio sample, downloading from S3 if needed."""
	try:
		# VoviAI Dataset uses 'audioLink' field for S3 URLs
		if "audioLink" not in sample:
			logger.error(f"No audioLink field found in sample: {sample.keys()}")
			return None
		
		# Download from S3 URL
		audio_path = download_audio_from_s3(sample["audioLink"])
		if not audio_path:
			logger.error(f"Failed to download audio from: {sample['audioLink']}")
			return None
			
		audio_data = load_audio_from_path(audio_path)
		if audio_data is None:
			logger.error(f"Failed to load audio from path: {audio_path}")
			return None
			
		# Get text transcription (VoviAI Dataset uses 'text' field)
		text = sample.get("text", "")
		if not text:
			logger.warning(f"Empty text found for audio: {sample['audioLink']}")
			return None
		
		return {
			"audio": audio_data,
			"text": normalize_text(text),
			"path": audio_data.get("path", "unknown"),
			# Keep additional metadata for potential use
			"speakerId": sample.get("speakerId", ""),
			"environment": sample.get("environment", ""),
			"gender": sample.get("gender", ""),
			"province_name": sample.get("province_name", ""),
			"region": sample.get("region", "")
		}
	except Exception as e:
		logger.error(f"Error processing audio sample with audioLink {sample.get('audioLink', 'unknown')}: {str(e)}")
		return None