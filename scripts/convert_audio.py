# scripts/convert_audio.py
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.audio_utils import convert_folder_to_wav  # Adjust the import path if necessary

def convert_audio_files(input_dir, output_dir):
    """Converts all audio files in the input directory to WAV format and saves them in the output directory."""
    convert_folder_to_wav(input_dir, output_dir)
    print(f"Converted all audio files in {input_dir} to WAV format in {output_dir}")

# Example usage: python scripts/convert_audio.py input_dir output_dir
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert audio files to WAV format.")
    parser.add_argument('input_dir', type=str, help="Directory containing audio files to convert.")
    parser.add_argument('output_dir', type=str, help="Directory to save converted WAV files.")

    args = parser.parse_args()
    convert_audio_files(args.input_dir, args.output_dir)