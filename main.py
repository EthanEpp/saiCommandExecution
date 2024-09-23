# import torch
# from src.models.speech_to_text import SpeechToText
# from src.services.cnet_inference import run_inference
# from src.models import cnet

# def main():
#     model_path = "./models/ctranfinal_bert_in_model_v1"
#     bert_addr = './bert_models/bert-large-uncased'
#     model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)
#     if torch.cuda.is_available():
#         model.cuda()
#     else:
#         print('You are NOT using cuda! Some problems may occur.')

#     stt = SpeechToText()
#     stt.start_microphone_stream()

#     try:
#         # while True:
#         print("Listening...")
#         user_speech = stt.process_audio_stream(seconds=5)  # Adjust duration as needed
#         print("Heard:", user_speech)
#         if user_speech:
#             closest_command = run_inference(user_speech, model)
#             print("Interpreted command:", closest_command)
#     except KeyboardInterrupt:
#         print("Stopping...")
#     finally:
#         stt.stop_microphone_stream()

# if __name__ == "__main__":
#     # main_text_only()
#     main()


import torch
from src.services.cnet_inference import run_inference
from src.models import cnet

def main():
    model_path = "/root/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/processor_model_08_11/ctranfinal_eic_8_27_V1"
    bert_addr = '/root/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/bert-large-uncased'
    model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('You are NOT using cuda! Some problems may occur.')
    user_speech = "Start a 5 minute timer and name it grimace"
    print(run_inference(user_speech, model))

if __name__ == "__main__":
    # main_text_only()
    model_path = "/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/saiCommandProcessor/processor_model_08_11/ctranfinal_eic_8_27_V1"
    bert_addr = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/saiCommandProcessor/bert-large-uncased'
    model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('You are NOT using cuda! Some problems may occur.')
    user_speech = "Start a 5 minute timer and name it grimace"
    # print(run_inference(user_speech, model))
    # print(run_inference("hide the clock", model))
    # print(run_inference("set volume to twenty two", model))

    print(run_inference("start a 5 minute timer named cooking and then can you look up how to change a lightbulb in a ceiling fan", model))
    print(run_inference("start a 5 minute timer named cooking then can you look up how to change a lightbulb in a ceiling fan", model))

    print(run_inference("set the system volume to 5 and open the spotify app", model))
    print(run_inference("turn on the room lights", model))

    # print(run_inference("how do i change a lightbulb in a ceiling fan", model))
    # print(run_inference("look up how to change a lightbulb in a ceiling fan", model))

    

