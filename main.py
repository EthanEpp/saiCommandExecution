
import psutil
import torch
from src.services.cnet_inference import run_inference
from src.models import cnet
import torch

import torch
print(f"PyTorch is using {torch.get_num_threads()} threads.")
torch.set_num_threads(1)  # Or adjust to a reasonable number based on available cores


# Function to monitor CPU usage
def log_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)  # Measure CPU usage every second
    print(f"CPU Usage: {cpu_usage}%")

if __name__ == "__main__":
    model_path = "/home/devel/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/processor_model_01_13_25/ctranfinal_revised_01_10_new_datav3"
    bert_addr = '/home/devel/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/bert-large-uncased-temp'
    bert_onnx_path = '/home/devel/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/bert_layer.onnx'
    model_path = "/home/devel/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/processor_model_01_13_25/ctranfinal_revised_01_10_new_datav3"
    bert_addr = '/home/devel/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/bert-large-uncased-temp'
    log_cpu_usage()
    # model = cnet.CNet(model_path=model_path, bert_addr=bert_addr)


    # log_cpu_usage()
    bert_onnx_path = '/home/devel/ai_test_eebs/EEcommandProcessor/saiCommandExecution/saiCommandProcessor/bert_layer.onnx'
    model = cnet.CNetOnnx(model_path=model_path, bert_onnx_path=bert_onnx_path, bert_addr=bert_addr)
    import torch.autograd.profiler as profiler

    with profiler.profile(record_shapes=True) as prof:
        # Your inference code here
        run_inference("Start a timer", model)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    

    if torch.cuda.is_available():
        model.cuda()
    else:
        print('You are NOT using cuda! Some problems may occur.')

    # Run inference with monitoring
    for sentence in [
        "Start a Cement Cure timer for 5 minutes",
        "Show my last patient schedule",
        "start a five minute timer make it the color green and name it avocado",
        "start a 5 minute timer named cooking and then can you look up how to change a lightbulb in a ceiling fan",
        "set the system volume to 5",
        "turn on the room lights",
        "how do i change a lightbulb in a ceiling fan"
    ]:
        print(run_inference(sentence, model))
        log_cpu_usage()
