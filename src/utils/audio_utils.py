import os
from pydub import AudioSegment

def convert_to_wav(source_path, target_path=None):
    """
    Convert an audio file to WAV format.

    Args:
        source_path (str): The file path of the source audio.
        target_path (str): Optional. The file path where the WAV file will be saved.

    Returns:
        str: The file path of the converted WAV file.
    """
    source_path = str(source_path)  # Convert to string if it's a PosixPath object
    audio_format = source_path.split('.')[-1]
    audio = AudioSegment.from_file(source_path, format=audio_format)
    if target_path is None:
        target_path = source_path.replace(f".{audio_format}", ".wav")
    audio.export(target_path, format="wav")
    return target_path

def convert_folder_to_wav(source_folder, target_folder):
    """
    Convert all audio files in a folder to WAV format and save them to another folder.

    Args:
        source_folder (str): The path to the folder containing the source audio files.
        target_folder (str): The path to the folder where the WAV files will be saved.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        if os.path.isfile(source_path):
            target_path = os.path.join(target_folder, os.path.splitext(filename)[0] + '.wav')
            convert_to_wav(source_path, target_path)
