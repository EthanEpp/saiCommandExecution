import torch
from src.services.cnet_inference import run_inference
from src.models import cnet
from transformers import BertModel
import torch
import torch.nn as nn

class BertLayerWrapper(nn.Module):
    def __init__(self, bert_model):
        super(BertLayerWrapper, self).__init__()
        self.bert_model = bert_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_encodings = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return bert_encodings['last_hidden_state'], bert_encodings['pooler_output']



bert_addr = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/saiCommandProcessor/bert-large-uncased-temp'
bert_model = BertModel.from_pretrained(bert_addr)

# Initialize the wrapper
bert_layer = BertLayerWrapper(bert_model)
bert_layer.eval()  # Set to evaluation mode

# Dummy inputs for ONNX export
batch_size = 1  # Adjust as needed
seq_length = 60  # Padded sequence length
dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length))  # Replace 30522 with your vocab size
dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int64)
dummy_token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.int64)

# Export to ONNX
onnx_path = "bert_layer.onnx"
torch.onnx.export(
    bert_layer,
    (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
    onnx_path,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "token_type_ids": {0: "batch_size", 1: "seq_length"},
        "last_hidden_state": {0: "batch_size", 1: "seq_length"},
        "pooler_output": {0: "batch_size"},
    },
    opset_version=14,  # Ensure this is supported by TensorRT
)
print(f"ONNX model saved to {onnx_path}")
