from src.models.speech_to_text import SpeechToText
from src.services.command_processor import CommandProcessor
import json

def load_commands():
    with open('data/commands.json', 'r') as file:
        data = json.load(file)
    return data.get('commands', [])


def main_text_only():
    commands = load_commands()
    processor = CommandProcessor(commands)

    # Example user input
    # user_speech = "Send a text message to Dr. Matt Wood saying the patient is ready for you in OR 1"  # Assume this is output from Whisper
    user_speech = "Can you search for the weather in Santa Barbara today"  # Assume this is output from Whisper
    # user_speech = "Can you search for the yen to USD conversion"  # Assume this is output from Whisper
    # user_speech = "Can you start a seven and a half minute timer"  # Assume this is output from Whisper
    # user_speech = "Begin a 10 minute timer and name it Blood Transfusion"  # Assume this is output from Whisper
    command = processor.find_closest_command(user_speech)
    print("Interpreted command:", command.command_type)
    print("Original input:", command.original_input)
    print("Preprocessed input:", command.preprocessed_input)
    print("Entities:", command.entities)
    print("Clauses:", command.clauses)


def main():
    commands = load_commands()
    processor = CommandProcessor(commands = commands, threshold=0.6)
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
    main_text_only()
    # main()


# # Example usage
# if __name__ == "__main__":
#     commands = ["set a timer", "google search"]
#     processor = CommandProcessor(commands)
#     user_input = "Set a timer for 5 minutes"
#     command, entities = processor.find_closest_command(user_input)
#     print(f"Command: {command}, Entities: {entities}")
