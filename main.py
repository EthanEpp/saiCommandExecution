import torch
from src.models.speech_to_text import SpeechToText
from src.services.cnet_inference import run_inference
from src.models import cnet

def main():
    model_path = "./models/ctranfinal_bert_in_model_v1"
    bert_addr = './bert_models/bert-large-uncased'
    model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('You are NOT using cuda! Some problems may occur.')

    stt = SpeechToText()
    stt.start_microphone_stream()

    try:
        # while True:
        print("Listening...")
        user_speech = stt.process_audio_stream(seconds=5)  # Adjust duration as needed
        print("Heard:", user_speech)
        if user_speech:
            closest_command = run_inference(user_speech, model)
            print("Interpreted command:", closest_command)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stt.stop_microphone_stream()

if __name__ == "__main__":
    # main_text_only()
    main()
