import pytest
import os
from pydub import AudioSegment
from src.utils.audio_utils import convert_to_wav, convert_folder_to_wav

@pytest.fixture
def setup_test_files(tmp_path):
    # Create some test audio files in the temporary directory
    file_formats = ["mp3", "m4a", "wav"]
    test_files = []
    for i, format in enumerate(file_formats):
        file_path = tmp_path / f"test{i}.{format}"
        audio = AudioSegment.silent(duration=1000)  # 1 second of silence
        if format == "m4a":
            audio.export(file_path, format="mp4", codec="aac")  # Use mp4 container for m4a
        else:
            audio.export(file_path, format=format)
        test_files.append(file_path)
    return test_files

def test_convert_to_wav(setup_test_files):
    for source_path in setup_test_files:
        if source_path.suffix != ".wav":
            target_path = str(source_path).replace(source_path.suffix, ".wav")
            result_path = convert_to_wav(source_path)
            assert result_path == target_path
            assert os.path.exists(result_path)

def test_convert_folder_to_wav(setup_test_files, tmp_path):
    source_folder = tmp_path
    target_folder = tmp_path / "wav_files"
    
    convert_folder_to_wav(source_folder, target_folder)
    
    for source_path in setup_test_files:
        if source_path.suffix != ".wav":
            target_path = target_folder / (source_path.stem + ".wav")
            assert os.path.exists(target_path)

if __name__ == "__main__":
    pytest.main()
