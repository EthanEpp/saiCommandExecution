# Speech Command System

This repository contains a speech command system that processes audio inputs, transcribes them into text, and executes corresponding commands. The system leverages various libraries for audio processing, speech-to-text conversion, and command execution.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [CustomGPT] (#customgpt)

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

## File Descriptions

`main.py`\
This is the entry point of the application. It initializes the necessary components and starts the main loop for listening and processing audio inputs.

`whisper_test.py`\
Contains test cases for validating the functionality of the Whisper model used for speech-to-text conversion.

`speech_to_text.py`\
Handles the conversion of speech to text using the Whisper model from OpenAI. This module includes functions to capture audio and process it into text format.

`command_processor.py`\
Processes the transcribed text to identify and execute corresponding commands. It contains the logic for parsing commands and triggering appropriate actions.

`audio_utils.py`\
Utility functions for handling audio data. This includes functions for recording audio, normalizing audio signals, and converting between different audio formats.

`embeddings.py`\
Manages the embeddings for text data to facilitate command recognition. It uses the `sentence_transformers` library to create and compare text embeddings.

`requirements.txt`\
Lists all the dependencies required to run the project, but key ones to note and understand are:

```plaintext
numpy==2.0.0
openai_whisper==20231117
PyAudio==0.2.14
pydub==0.25.1
scipy==1.13.1
sentence_transformers==3.0.1
torch==2.3.1
transformers==4.41.2
```
You can install all dependencies using the provided `requirements.txt` file.


## CustomGPT

For questions or assistance regarding this codebase, [click here](https://chatgpt.com/g/g-ZEvSUqsh0-speech-commands-expert) to access a customGPT instance that has been trained to assist with any inquiries about the speech command system and has access to the codebase. Note that this should only be used for reference and may not automatically be up to date with the current version of the code. Last customGPT knowledge update: June 24, 2024
