import torch
from src.services.cnet_inference import run_inference
from src.models import cnet

<<<<<<< Updated upstream
def main():
    model_path = "/root/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/processor_model_08_11/ctranfinal_eic_8_27_V1"
    bert_addr = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/bert-large-uncased-temp'
    model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('You are NOT using cuda! Some problems may occur.')
    user_speech = "Start a 5 minute timer and name it grimace"
    print(run_inference(user_speech, model))
=======
# import onnx
# bert_onnx = "/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/bert_layer_fixed.onnx"
# onnx_model = onnx.load(bert_onnx)
# for input in onnx_model.graph.input:
#     print(f"Input Name: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
>>>>>>> Stashed changes

    
if __name__ == "__main__":
    # main_text_only()
    model_path = "/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/saiCommandProcessor/processor_model_01_13_25/ctranfinal_revised_01_10_new_datav3"
<<<<<<< Updated upstream
    bert_addr = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/bert-large-uncased-temp'
    model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)
=======
    # model_path = "/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/saiCommandProcessor/processor_model_10_24/ctranfinal_revised_10_24_batch_4"
    bert_addr = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/saiCommandProcessor/bert-large-uncased-temp'
    # model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)

    bert_onnx_path = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/bert_layer_fixed.onnx'
    model = cnet.CNetOnnx(model_path=model_path, bert_onnx_path=bert_onnx_path, bert_addr=bert_addr)
>>>>>>> Stashed changes
    if torch.cuda.is_available():
        model.cuda()
    else:
        print('You are NOT using cuda! Some problems may occur.')
    user_speech = "Start a 5 minute timer and name it grimace"
    # print(run_inference(user_speech, model))
    # print(run_inference("hide the clock", model))
    # print(run_inference("set volume to twenty two", model))

<<<<<<< Updated upstream
    print(run_inference("Start a Cement Cure timer for 5 minutes", model))
    print(run_inference("Show my last patient schedule", model))
    # print(run_inference("start a 5 minute timer named cooking then can you look up how to change a lightbulb in a ceiling fan", model))

    # print(run_inference("set the system volume to 5 and open the spotify app", model))
    # print(run_inference("turn on the room lights", model))

    # print(run_inference("how do i change a lightbulb in a ceiling fan", model))
    # print(run_inference("look up how to change a lightbulb in a ceiling fan", model))

    
=======
    # print(run_inference("Start a Cement Cure timer for 5 minutes", model))
    # print(run_inference("Show my last patient schedule", model))
    # print(run_inference("start a five minute timer make it the color green and name it avocado", model))
    print(run_inference("Accept call", model))
    print(run_inference("Accept the call", model))
>>>>>>> Stashed changes
