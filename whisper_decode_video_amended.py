import os
import argparse
import numpy as np
import torch
import whisper
from whisper.tokenizer import get_tokenizer
from utils import load_video_feats  # Assuming this is still needed
import librosa  # For loading audio
print(f"NumPy version: {np.__version__}")
print(f"np.float exists: {hasattr(np, 'float')}")
# Constants
SAMPLE_RATE = 16000

def main():
    parser = argparse.ArgumentParser(description="Decode audio and video files using Whisper for AVSR.")
    parser.add_argument('--lang', default='en', type=str, help='Language for decoding (e.g., "en" for English)')
    parser.add_argument('--model-type', default='small', type=str, help='Whisper model size (e.g., small)')
    parser.add_argument('--audio', required=True, type=str, help='Path to the input audio file')
    parser.add_argument('--video', required=True, type=str, help='Path to the input video file')
    parser.add_argument('--whisper-path', default="models/", type=str, help='Path to OpenAI Whisper weights')
    parser.add_argument('--av-hubert-ckpt', default="models/large_noise_pt_noise_ft_433h_only_weights.pt", 
                        type=str, help='Path to AV-Hubert checkpoint')
    parser.add_argument('--av-hubert-path', default="av_hubert/avhubert/", type=str, 
                        help='Path to AV-Hubert code directory')
    parser.add_argument('--checkpoint-path', default=None, type=str, help='Path to custom model checkpoint')
    parser.add_argument('--fp16', default=0, type=int, choices=[0, 1], help='Use FP16 (1) or CPU (0)')
    
    args = parser.parse_args()

    # Validate file paths
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")

    # Determine if the model is multilingual
    multilingual = True if 'large' in args.model_type or 'en' not in args.model_type else False
    tokenizer = get_tokenizer(multilingual=multilingual, task="transcribe")

    # Load the Whisper model
    whisper_model = whisper.load_model(
        args.model_type,
        download_root=args.whisper_path,
        video=True,
        video_model_path=args.av_hubert_ckpt,
        av_hubert_path=args.av_hubert_path,
        av_hubert_encoder=0,
        av_fusion="",
        add_gated_x_attn=0
    )

    # Load checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
        state_dict_updated = {k[6:]: v for k, v in state_dict.items()}
        whisper_model.load_state_dict(state_dict_updated, strict=False)

    # Set decoding options
    options = whisper.DecodingOptions(
        task="transcribe",
        language=args.lang,
        fp16=bool(args.fp16),
        without_timestamps=True,
        beam_size=1
    )

    # Load audio and convert to mel spectrogram
    audio, sr = librosa.load(args.audio, sr=SAMPLE_RATE)
    audio = torch.from_numpy(audio).float()  # Convert to tensor
    mel = whisper.log_mel_spectrogram(audio, n_mels=80)  # Generate mel spectrogram with 80 bands
    mel = mel.unsqueeze(0)  # Add batch dimension: [1, 80, time]

    # Load video data
    video = load_video_feats(args.video, train=False)

    # Perform decoding
    result = whisper_model.decode(mel, options, video)
    transcription = result.text.strip()

    # Output transcription
    print(transcription)

if __name__ == "__main__":
    main()