import sys
import os
import numpy as np
import soundfile as sf
from Exceptions import WeightDownladException
from downloader.SampleDownloader import downloadInitialSamples
from downloader.WeightDownloader import downloadWeight
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.VoiceChangerManager import VoiceChangerManager
from mods.log_control import VoiceChangaerLogger
from scipy import signal

CONFIG = {
    "model_dir": "model_dir",
    "sample_mode": "production",
    "content_vec_500": "pretrain/checkpoint_best_legacy_500.pt",
    "content_vec_500_onnx": "pretrain/content_vec_500.onnx",
    "content_vec_500_onnx_on": True,
    "hubert_base": "pretrain/hubert_base.pt",
    "hubert_base_jp": "pretrain/rinna_hubert_base_jp.pt",
    "hubert_soft": "pretrain/hubert/hubert-soft-0d54a1f4.pt",
    "whisper_tiny": "pretrain/whisper_tiny.pt",
    "nsf_hifigan": "pretrain/nsf_hifigan/model",
    "crepe_onnx_full": "pretrain/crepe_onnx_full.onnx",
    "crepe_onnx_tiny": "pretrain/crepe_onnx_tiny.onnx",
    "rmvpe": "pretrain/rmvpe.pt",
    "rmvpe_onnx": "pretrain/rmvpe.onnx",
}

# Initialize logger
VoiceChangaerLogger.get_instance().initialize(initialize=True)
logger = VoiceChangaerLogger.get_instance().getLogger()
logger.debug(f"---------------- Initializing Voice Changer -----------------")

def initialize_voice_changer():
    """Initialize and return a voice changer instance with the configured parameters."""
    # Set up voice changer parameters
    voice_changer_params = VoiceChangerParams(
        model_dir=CONFIG["model_dir"],
        content_vec_500=CONFIG["content_vec_500"],
        content_vec_500_onnx=CONFIG["content_vec_500_onnx"],
        content_vec_500_onnx_on=CONFIG["content_vec_500_onnx_on"],
        hubert_base=CONFIG["hubert_base"],
        hubert_base_jp=CONFIG["hubert_base_jp"],
        hubert_soft=CONFIG["hubert_soft"],
        nsf_hifigan=CONFIG["nsf_hifigan"],
        crepe_onnx_full=CONFIG["crepe_onnx_full"],
        crepe_onnx_tiny=CONFIG["crepe_onnx_tiny"],
        rmvpe=CONFIG["rmvpe"],
        rmvpe_onnx=CONFIG["rmvpe_onnx"],
        sample_mode=CONFIG["sample_mode"],
        whisper_tiny=CONFIG["whisper_tiny"],
    )
    
    # Set up voice changer params manager
    vcparams = VoiceChangerParamsManager.get_instance()
    vcparams.setParams(voice_changer_params)
    
    # Download required weights
    try:
        downloadWeight(voice_changer_params)
    except WeightDownladException:
        logger.error("Failed to download weight for rvc")
        raise

    # Download initial samples
    try:
        downloadInitialSamples(CONFIG["sample_mode"], CONFIG["model_dir"])
    except Exception as e:
        logger.error(f"Loading sample failed: {e}")
        raise

    # Initialize voice changer manager
    voice_changer_manager = VoiceChangerManager.get_instance(voice_changer_params)
    return voice_changer_manager

def process_audio_file(voice_changer: VoiceChangerManager, input_file: str, output_file: str, chunk_size: int = 4096):
    """Process an audio file through the voice changer in chunks.
    
    Args:
        voice_changer: Initialized voice changer instance
        input_file: Path to input audio file
        output_file: Path to save the processed audio
        chunk_size: Size of audio chunks to process at once
    """
    # Read the input audio file
    audio_data, sample_rate = sf.read(input_file)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Ensure sample rate matches the voice changer's expected rate (48000Hz)
    if sample_rate != 48000:
        # Calculate number of samples for target length
        target_length = int(len(audio_data) * 48000 / sample_rate)
        # Resample audio
        audio_data = signal.resample(audio_data, target_length)
        sample_rate = 48000
    
    # Convert to int16 with proper scaling
    audio_data = (audio_data * 32768.0).astype(np.int16)
    
    # Process audio in chunks
    processed_chunks = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        # Pad the last chunk if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        print(f"processing sample {i} to {i + chunk_size}")
        # Process through voice changer
        processed_chunk, _ = voice_changer.changeVoice(chunk)
        processed_chunks.append(processed_chunk)
    
    # Combine processed chunks
    processed_audio = np.concatenate(processed_chunks)
    
    # Convert back to float32 for saving (proper scaling)
    processed_audio = processed_audio.astype(np.float32) / 32768.0
    
    # Save the processed audio
    sf.write(output_file, processed_audio, sample_rate)
    logger.info(f"Processed audio saved to {output_file}")

if __name__ == "__main__":
    try:
        # Initialize the voice changer
        voice_changer = initialize_voice_changer()
        logger.info("Voice changer initialized successfully")
        voice_changer.update_settings("modelSlotIndex", 4)
        voice_changer.update_settings("tran", 6)
        
        # Example usage with an audio file
        input_file = "test.wav"  # Replace with your input file
        output_file = "test_output.wav"  # Where to save the processed audio
        
        # Process the audio file
        process_audio_file(voice_changer, input_file, output_file)
        
    except Exception as e:
        logger.error(f"Failed to process audio: {e}")
