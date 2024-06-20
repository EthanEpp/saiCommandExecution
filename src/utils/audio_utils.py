from pydub import AudioSegment
import os

def convert_to_wav(source_path, target_path=None):
    """
    Convert an audio file to WAV format.

    Args:
        source_path (str): The file path of the source audio.
        target_path (str): Optional. The file path where the WAV file will be saved.

    Returns:
        str: The file path of the converted WAV file.
    """
    audio_format = source_path.split('.')[-1]
    audio = AudioSegment.from_file(source_path, format=audio_format)
    if target_path is None:
        target_path = source_path.rsplit('.', 1)[0] + '.wav'
    audio.export(target_path, format='wav')
    return target_path

def convert_folder_to_wav(source_folder, target_folder=None):
    """
    Convert all audio files in a folder to WAV and save them in the target folder.

    Args:
        source_folder (str): The directory containing audio files to convert.
        target_folder (str): The directory where the converted WAV files should be saved.
                             If not provided, source_folder with '_wav' suffix will be used.
    """
    if target_folder is None:
        target_folder = source_folder + "_wav"

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if not filename.endswith('.wav'):  # Avoid re-converting WAV files
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, os.path.splitext(filename)[0] + '.wav')
            convert_to_wav(source_path, target_path)

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        source_folder = sys.argv[1]
        convert_folder_to_wav(source_folder)
        print(f"All audio files in {source_folder} have been converted to WAV and saved in {source_folder}_wav.")
    else:
        print("Please provide the path to the folder containing audio files.")
