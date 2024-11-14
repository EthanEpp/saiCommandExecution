[![CI](https://github.com/EthanEpp/saiCommandExecution/actions/workflows/ci.yml/badge.svg)](https://github.com/EthanEpp/saiCommandExecution/actions/workflows/ci.yml)

# Speech Command System

This repository contains a speech command system that processes audio inputs, transcribes them into text, and executes corresponding commands. The system utilizers whisber and a distilled SBERT paraphrase model.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Notebooks](#notebooks)
- [Modules](#modules)
- [Scripts](#scripts)
- [Tests](#tests)
- [Continuous Integration](#continuous-integration)
- [Contact](#contact)

## Installation

To get started with the speech command system, clone the repository and install the required dependencies:

```bash
git clone https://github.com/EthanEpp/saiCommandExecution/
git clone https://huggingface.co/ethan3048/saiCommandProcessor
cd ./saiCommandExecution
pip install -r requirements.txt
```

## Usage

Run the main script to start the system:

```bash
python main.py
```

This will initialize the system, start listening for 5 seconds for a command, then output the transcribed audio and the mapped command.


## Dependencies

- numpy==1.26.4
- openai_whisper==20231117
- PyAudio==0.2.14
- pydub==0.25.1
- pytest==8.2.2
- sacremoses==0.1.1
- scipy==1.13.1
- scikit_learn==1.4.2
- sentence_transformers==3.0.1
- sounddevice==0.4.7
- spacy==3.7.5
- torch==2.3.1
- tqdm==4.66.4
- transformers==4.41.2

## Notebooks

### Command Pipeline (`notebooks/ctranCommandPipeline.ipynb`)

This notebook is designed to be run in colab, and goes through model loading and inference either through microphone input that is transcribed by whisper or direct text input.

## Modules

### Speech-to-Text (`src/models/speech_to_text.py`)

This module converts spoken language into text using the Whisper model from OpenAI.

### Cnet (`src/models/cnet.py`)

This module contains the architecture for the slot filling and intent detection that maps transcribed speech inputs to their corresponding command and necessary tags.

### Audio Utilities (`src/utils/audio_utils.py`)

This module contains utility functions for handling audio files such as converting them to wav.

### CNET Inference (`src/services/cnet_inference.py`)

This module contains the inference function for running inference on a model.

## Scripts

### Convert Audio (`scripts/convert_audio.py`)

This script converts audio files to wav.
Example usage in home directory: `python scripts/convert_audio.py input_dir output_dir`

### Whisper Test (`scripts/whisper_test.py`)

This script is used for testing whisper to make sure the model and dependencies are loaded correctly.
Example usage in home directory: `python scripts/whisper_test.py audio_files`

### Train New Model (`scripts/train_new_model`)

This script is used for when you would like to train a new CNET model.

## Tests

The repository includes unit and integration tests to ensure the functionality of the modules.

### Unit Tests

- `tests/test_speech_to_text_unit.py`: Tests for the speech-to-text module.
- `tests/test_audio_utils.py`: Tests for the audio utilities module.

### Integration Tests

- `tests/test_speech_to_text_integration.py`: Integration tests for the speech-to-text module.


## Continuous Integration

The project uses GitHub Actions for continuous integration. The configuration file is located at [`.github/workflows/ci.yml`](https://github.com/EthanEpp/saiCommandExecution/blob/main/.github/workflows/ci.yml).


## Contact

For any questions or suggestions, please contact Ethan Epp at [eepp@ucsb.edu].
