# Speech Command System

This repository contains a speech command system that processes audio inputs, transcribes them into text, and executes corresponding commands. The system utilizers whisber and a distilled SBERT paraphrase model.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Modules](#modules)
- [Scripts](#scripts)
- [Tests](#tests)
- [Continuous Integration](#continuous-integration)
- [CustomGPT](#customgpt)

## Installation

To get started with the speech command system, clone the repository and install the required dependencies:

```bash
git clone https://github.com/EthanEpp/saiCommandExecution
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

- `numpy==2.0.0`
- `openai_whisper==20231117`
- `PyAudio==0.2.14`
- `pydub==0.25.1`
- `pytest==8.2.2`
- `scipy==1.14.0`
- `sentence_transformers==3.0.1`
- `torch==2.3.1`
- `transformers==4.41.2`


## Modules

### Speech-to-Text (`src/models/speech_to_text.py`)

This module converts spoken language into text using the Whisper model from OpenAI.

### Embeddings (`src/models/embeddings.py`)

This module generates embeddings for text inputs to facilitate natural language understanding using a paraphrase SBERT.

### Command Processor (`src/services/command_processor.py`)

This module processes the commands derived from speech inputs and determines their corrosponding command action.

### Audio Utilities (`src/utils/audio_utils.py`)

This module contains utility functions for handling audio files such as converting them to wav.

## Scripts

### Convert Audio (`scripts/convert_audio.py`)

This script converts audio files to wav.

### Whisper Test (`scripts/whisper_test.py`)

This script is used for testing whisper to make sure the model and dependencies are loaded correctly.



## Tests

The repository includes unit and integration tests to ensure the functionality of the modules.

### Unit Tests

- `tests/test_embeddings_unit.py`: Tests for the embeddings module.
- `tests/test_speech_to_text_unit.py`: Tests for the speech-to-text module.
- `tests/test_command_processor_unit.py`: Tests for the command processor module.
- `tests/test_audio_utils.py`: Tests for the audio utilities module.

### Integration Tests

- `tests/test_embeddings_integration.py`: Integration tests for the embeddings module.
- `tests/test_speech_to_text_integration.py`: Integration tests for the speech-to-text module.
- `tests/test_command_processor_integration.py`: Integration tests for the command processor module.


## Continuous Integration

The project uses GitHub Actions for continuous integration. The configuration file is located at `.github/workflows/ci.yml`.

## CustomGPT

For questions or assistance regarding this codebase, [click here](https://chatgpt.com/g/g-ZEvSUqsh0-speech-commands-expert) to access a customGPT instance that has been trained to assist with any inquiries about the speech command system and has access to the codebase. Note that this should only be used for reference and may not automatically be up to date with the current version of the code. Last customGPT knowledge update: June 24, 2024
