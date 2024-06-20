from src.models.speech_to_text import SpeechToText
from src.services.command_processor import CommandProcessor
import json

def load_commands():
    with open('data/commands.json', 'r') as file:
        data = json.load(file)
        return data['commands']


# def main():
#     commands = load_commands()
#     processor = CommandProcessor(commands)

#     # Example user input
#     user_speech = "Text Matt and tell him the patient is ready"  # Assume this is output from Whisper
#     closest_command = processor.find_closest_command(user_speech)
#     print("Interpreted command:", closest_command)



def main():
    commands = load_commands()
    processor = CommandProcessor(commands)
    stt = SpeechToText()
    stt.start_microphone_stream()

    try:
        # while True:
        print("Listening...")
        user_speech = stt.process_audio_stream(seconds=5)  # Adjust duration as needed
        print("Heard:", user_speech)
        if user_speech:
            closest_command = processor.find_closest_command(user_speech)
            print("Interpreted command:", closest_command)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stt.stop_microphone_stream()

if __name__ == "__main__":
    main()
