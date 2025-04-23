import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import pickle
from transformers import BertModel, BertTokenizer
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
from src.models.cnet import generate_square_subsequent_mask, generate_square_diagonal_mask, load_mapping, PositionalEncoding


# cuda.init()

class BertLayerTRT:
    def __init__(self, trt_engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        # Load TensorRT Engine
        with open(trt_engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Get Tensor Names & Print Bindings Info
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        for i, tensor_name in enumerate(tensor_names):
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = self.engine.get_tensor_dtype(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            location = "GPU" if self.engine.get_tensor_location(tensor_name) == trt.TensorLocation.DEVICE else "CPU"

            # print(f"Binding {i} - Name: {tensor_name}, IsInput: {is_input}, Shape: {shape}, Dtype: {dtype}, Location: {location}")
        self.stream = cuda.Stream()

        # Retrieve tensor names
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

        # Allocate device buffers for inputs and outputs
        self.device_buffers = {}
        self.shapes = {}
        self.dtypes = {}
        self.bindings = []

        for tensor_name in self.tensor_names:
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(tensor_name)))
            device_mem = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize)

            self.device_buffers[tensor_name] = device_mem
            self.shapes[tensor_name] = shape
            self.dtypes[tensor_name] = dtype
            self.bindings.append(int(device_mem))

            # Debugging
            # print(f"Allocated buffer for {tensor_name}: Shape={shape}, Dtype={dtype}, Location=GPU")

    def forward(self, bert_tokens, bert_mask, bert_tok_typeid):
        # Extract batch_size and sequence_length from the input tensors
        batch_size, sequence_length = bert_tokens.shape  # Example: (1, 60)

        # Set the TensorRT engine’s input shape dynamically
        self.context.set_input_shape("input_ids", (batch_size, sequence_length))
        self.context.set_input_shape("attention_mask", (batch_size, sequence_length))
        self.context.set_input_shape("token_type_ids", (batch_size, sequence_length))

        # Convert inputs to NumPy arrays with correct shapes and dtypes
        input_dict = {
            "input_ids": bert_tokens.cpu().numpy().astype(self.dtypes["input_ids"]),
            "attention_mask": bert_mask.cpu().numpy().astype(self.dtypes["attention_mask"]),
            "token_type_ids": bert_tok_typeid.cpu().numpy().astype(self.dtypes["token_type_ids"]),
        }

        # Debugging: Print input tensor shapes and expected binding shapes
        for name, data in input_dict.items():
            print(f" - {name}: Shape={data.shape}, Dtype={data.dtype}, Expected Shape={self.shapes[name]}")
        # Copy input data to GPU
        for name, data in input_dict.items():
            print(f"Copying {name} to GPU Buffer: {self.device_buffers[name]}")
            cuda.memcpy_htod(self.device_buffers[name], data)

        # Run inference
        self.context.execute_v2(self.bindings)
        # Retrieve outputs
        output_dict = {}
        for name in ["last_hidden_state", "pooler_output"]:
            output_array = np.empty(self.shapes[name], dtype=self.dtypes[name])
            cuda.memcpy_dtoh(output_array, self.device_buffers[name])
            output_dict[name] = torch.tensor(output_array).to(bert_tokens.device)

        return output_dict["last_hidden_state"], output_dict["pooler_output"]


class EncoderTRT:
    def __init__(self, trt_engine_path, device="cuda"):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load the TensorRT Engine
        with open(trt_engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Bindings
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.device_buffers = {}
        self.shapes = {}
        self.dtypes = {}
        self.bindings = []

        for name in self.tensor_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            device_mem = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize)

            self.device_buffers[name] = device_mem
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            self.bindings.append(int(device_mem))

    def forward(self, bert_last_hidden):
        encoder_start = time.time()
        batch_size, seq_len, hidden = bert_last_hidden.shape
        self.context.set_input_shape("bert_last_hidden", (batch_size, seq_len, hidden))

        # Convert input to correct format
        input_np = bert_last_hidden.cpu().numpy().astype(self.dtypes["bert_last_hidden"])
        cuda.memcpy_htod(self.device_buffers["bert_last_hidden"], input_np)
        encoder_copy_time = time.time()
        print("encoder copy time", encoder_copy_time -encoder_start)
        # Execute inference
        self.context.execute_v2(self.bindings)
        encoder_execute_time =time.time()
        print("encoder execute time", encoder_execute_time - encoder_copy_time)
        # Retrieve output
        output_np = np.empty(self.shapes["encoder_output"], dtype=self.dtypes["encoder_output"])
        cuda.memcpy_dtoh(output_np, self.device_buffers["encoder_output"])
        encoder_end_time = time.time()
        print("encoder copy time", (encoder_copy_time - encoder_start) + (encoder_end_time - encoder_execute_time))
        return torch.tensor(output_np, device=self.device)



class MiddleTRT:
    def __init__(self, trt_engine_path, device="cuda"):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load TensorRT Engine
        with open(trt_engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Setup bindings and allocate memory
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.device_buffers = {}
        self.shapes = {}
        self.dtypes = {}
        self.bindings = []

        for name in self.tensor_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            device_mem = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize)

            self.device_buffers[name] = device_mem
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            self.bindings.append(int(device_mem))

    def forward(self, fromencoder, input_masking):
        batch_size, seq_len, hidden = fromencoder.shape

        # Set input shapes
        self.context.set_input_shape("fromencoder", (batch_size, seq_len, hidden))
        self.context.set_input_shape("input_masking", (batch_size, seq_len))

        # Prepare input dict
        input_dict = {
            "fromencoder": fromencoder.cpu().numpy().astype(self.dtypes["fromencoder"]),
            "input_masking": input_masking.cpu().numpy().astype(self.dtypes["input_masking"])
        }

        # Copy inputs to device
        for name, data in input_dict.items():
            cuda.memcpy_htod(self.device_buffers[name], data)

        # Execute inference
        self.context.execute_v2(self.bindings)

        # Retrieve output
        output_array = np.empty(self.shapes["output"], dtype=self.dtypes["output"])
        cuda.memcpy_dtoh(output_array, self.device_buffers["output"])

        return torch.tensor(output_array, device=self.device)



class DecoderTRT(nn.Module):
    def __init__(self, trt_engine_path, device="cuda"):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        # Load the TensorRT engine
        with open(trt_engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.stream = cuda.Stream()

        # Prepare bindings
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.device_buffers = {}
        self.shapes = {}
        self.dtypes = {}
        self.bindings = []

        for name in self.tensor_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            device_mem = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize)

            self.device_buffers[name] = device_mem
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            self.bindings.append(int(device_mem))

    def forward(self, input_tensor, encoder_outputs, encoder_maskings, bert_subtoken_maskings=None):
        decoder_start_time = time.time()
        batch_size, seq_len, hidden = encoder_outputs.shape

        # Set input shapes dynamically
        self.context.set_input_shape("encoder_outputs", (batch_size, seq_len, hidden))
        self.context.set_input_shape("encoder_maskings", (batch_size, seq_len))

        # Copy input data
        cuda.memcpy_htod(self.device_buffers["encoder_outputs"], encoder_outputs.cpu().numpy().astype(self.dtypes["encoder_outputs"]))
        cuda.memcpy_htod(self.device_buffers["encoder_maskings"], encoder_maskings.cpu().numpy().astype(self.dtypes["encoder_maskings"]))
        decoder_copy_time = time.time()
        # Run inference
        self.context.execute_v2(self.bindings)
        decoder_execute_time = time.time()
        # Retrieve outputs
        slot_scores_np = np.empty(self.shapes["slot_scores"], dtype=self.dtypes["slot_scores"])
        intent_score_np = np.empty(self.shapes["intent_score"], dtype=self.dtypes["intent_score"])

        cuda.memcpy_dtoh(slot_scores_np, self.device_buffers["slot_scores"])
        cuda.memcpy_dtoh(intent_score_np, self.device_buffers["intent_score"])
        decoder_end_time = time.time()
        print("Decoder execute time", decoder_execute_time - decoder_start_time)
        return torch.tensor(slot_scores_np, device=self.device), torch.tensor(intent_score_np, device=self.device)




class CNetTRT(nn.Module):
    def __init__(self, model_path=None, bert_trt_path=None, encoder_trt_path=None, middle_trt_path=None,decoder_trt_path=None, bert_addr=None, padded_length=60):
        super(CNetTRT, self).__init__()
        self.length = padded_length
        self.bert_addr = bert_addr  # Keep for compatibility with tokenization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ✅ Load tokenizer only once
        try:
            self.tokenizer = BertTokenizer.from_pretrained(bert_addr)
        except Exception:
            # fallback if you're using torch.hub
            self.tokenizer = torch.hub.load(bert_addr, 'tokenizer', bert_addr, verbose=False, source="local")

        if model_path:
            self.word2index = load_mapping(f'{model_path}-word2index.pkl')
            self.index2word = load_mapping(f'{model_path}-index2word.pkl')
            self.tag2index = load_mapping(f'{model_path}-tag2index.pkl')
            self.index2tag = load_mapping(f'{model_path}-index2tag.pkl')
            self.intent2index = load_mapping(f'{model_path}-intent2index.pkl')
            self.index2intent = load_mapping(f'{model_path}-index2intent.pkl')
        else:
            print("No model path provided, this should only occur if you are training")
            self.word2index, self.index2word, self.tag2index, self.index2tag, self.intent2index, self.index2intent = word2index, index2word, tag2index, index2tag, intent2index, index2intent

        # Replace ONNX-based BertLayer with TensorRT version
        self.bert_layer = BertLayerTRT(bert_trt_path)
        self.encoder = EncoderTRT(encoder_trt_path)
        self.middle = MiddleTRT(middle_trt_path)
        self.decoder = DecoderTRT(decoder_trt_path)
        if model_path:
            import __main__
            __main__.PositionalEncoding = PositionalEncoding  # Ensure PositionalEncoding is available in the namespace


    def forward(self, bert_info, input, encoder_maskings, bert_subtoken_maskings=None, infer=False):
        # Process the input through the TensorRT-based BERT layer
        bert_tokens, bert_mask, bert_tok_typeid = bert_info
        start_time = time.time()
        bert_last_hidden, bert_pooler_output = self.bert_layer.forward(bert_tokens, bert_mask, bert_tok_typeid)
        end_time = time.time()
        print("total time bert layer", end_time -start_time)
        # Pass the BERT last hidden state through the encoder
        start_time_encoder = time.time()
        encoder_output = self.encoder.forward(bert_last_hidden)
        end_time_encoder = time.time()
        print("total time encoder", end_time_encoder - start_time_encoder)
        # Pass the encoder output through the middle component
        middle_output = self.middle.forward(encoder_output, encoder_maskings)
        end_time_middle = time.time()
        print("total time middle", end_time_middle - end_time_encoder)
        # Pass the middle output and other inputs through the decoder
        # slot_scores, intent_score = self.decoder(input, middle_output, encoder_maskings, self.tag2index, bert_subtoken_maskings, infer)
        slot_scores, intent_score = self.decoder(input, middle_output, encoder_maskings, bert_subtoken_maskings)
        end_time_decoder = time.time()
        print("total time decoder", end_time_decoder-end_time_middle)
        return slot_scores, intent_score
