import torch
from src.services.cnet_inference import run_inference
from src.models import cnet

def main():
    model_path = "/root/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/processor_model_08_11/ctranfinal_eic_8_27_V1"
    bert_addr = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/saiCommandProcessor/bert-large-uncased-temp'
    model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('You are NOT using cuda! Some problems may occur.')
    user_speech = "Start a 5 minute timer and name it grimace"
    print(run_inference(user_speech, model))

if __name__ == "__main__":
    # main_text_only()
    model_path = "/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/saiCommandProcessor/processor_model_01_13_25/ctranfinal_revised_01_10_new_datav3"
    bert_addr = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/saiCommandProcessor/bert-large-uncased-temp'
    # model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)

    bert_onnx_path = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/bert_layer.onnx'
    model = cnet.CNetOnnx(model_path=model_path, bert_onnx_path=bert_onnx_path, bert_addr=bert_addr)
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('You are NOT using cuda! Some problems may occur.')

    # print(run_inference("Start a Cement Cure timer for 5 minutes", model))
    print(run_inference("Show my last patient schedule", model))
    print(run_inference("start a five minute timer make it the color green and name it avocado", model))
    print(run_inference("Start a Cement Cure timer for 5 minutes", model))