import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import pickle
from transformers import BertModel
import tensorrt as trt

ENV_BERT_ID_CLS=False # use cls token for id classification
ENV_EMBEDDING_SIZE=1024 # dimention of embbeding, bertbase=768,bertlarge&elmo=1024
ENV_BERT_ADDR='/Users/SAI/Documents/Code/wakeWord/wakeWordForked/saiCommandExecution/bert-large-uncased-temp'
ENV_SEED=1331
ENV_CNN_FILTERS=128
ENV_CNN_KERNELS=4
ENV_HIDDEN_SIZE=ENV_CNN_FILTERS*ENV_CNN_KERNELS

# Define device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# generates transformer mask
def generate_square_subsequent_mask(sz: int) :
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
def generate_square_diagonal_mask(sz: int) :
    """Generates a matrix which there are zeros on diag and other indexes are -inf."""
    return torch.triu(torch.ones(sz,sz)-float('inf'), diagonal=1)+torch.tril(torch.ones(sz,sz)-float('inf'), diagonal=-1)

def load_mapping(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# positional embedding used in transformers
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# #start of the shared encoder
# class BertLayer(nn.Module):
#     def __init__(self, bert_addr):
#         super(BertLayer, self).__init__()
#         self.bert_model = torch.hub.load(bert_addr, 'model', bert_addr,source="local")
class BertLayer(nn.Module):
    def __init__(self, bert_addr):
        super(BertLayer, self).__init__()
        # Load the fine-tuned model
        self.bert_model = BertModel.from_pretrained(bert_addr)

    def forward(self, bert_info=None):
        (bert_tokens, bert_mask, bert_tok_typeid) = bert_info
        bert_encodings = self.bert_model(bert_tokens, bert_mask, bert_tok_typeid)
        bert_last_hidden = bert_encodings['last_hidden_state']
        bert_pooler_output = bert_encodings['pooler_output']
        return bert_last_hidden, bert_pooler_output

class BertLayerONNX:
    def __init__(self, onnx_model_path):
        # Load the ONNX model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


    def forward(self, bert_tokens, bert_mask, bert_tok_typeid):
        # Prepare inputs for ONNX Runtime
        inputs = {
            "input_ids": bert_tokens.cpu().numpy(),
            "attention_mask": bert_mask.cpu().numpy(),
            "token_type_ids": bert_tok_typeid.cpu().numpy(),
        }
        # Run inference
        outputs = self.session.run(["last_hidden_state", "pooler_output"], inputs)
        # Convert outputs back to PyTorch tensors
        bert_last_hidden = torch.tensor(outputs[0]).to(bert_tokens.device)
        bert_pooler_output = torch.tensor(outputs[1]).to(bert_tokens.device)
        return bert_last_hidden, bert_pooler_output




class Encoder(nn.Module):
    def __init__(self, p_dropout=0.5):
        super(Encoder, self).__init__()
        self.filter_number = ENV_CNN_FILTERS
        self.kernel_number = ENV_CNN_KERNELS  # tedad size haye filter : 2,3,5 = 3
        self.embedding_size = ENV_EMBEDDING_SIZE
        self.activation = nn.ReLU()
        self.p_dropout = p_dropout
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(2,),
                               padding="same", padding_mode="zeros")
        self.conv2 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(3,),
                               padding="same", padding_mode="zeros")
        self.conv3 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(5,),
                               padding="same", padding_mode="zeros")
        self.conv4 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(1,),
                               padding="same", padding_mode="zeros")

    def forward(self, bert_last_hidden):
        trans_embedded = torch.transpose(bert_last_hidden, dim0=1, dim1=2)
        convolve1 = self.activation(self.conv1(trans_embedded))
        convolve2 = self.activation(self.conv2(trans_embedded))
        convolve3 = self.activation(self.conv3(trans_embedded))
        convolve4 = self.activation(self.conv4(trans_embedded))
        convolve1 = torch.transpose(convolve1, dim0=1, dim1=2)
        convolve2 = torch.transpose(convolve2, dim0=1, dim1=2)
        convolve3 = torch.transpose(convolve3, dim0=1, dim1=2)
        convolve4 = torch.transpose(convolve4, dim0=1, dim1=2)
        output = torch.cat((convolve4, convolve1, convolve2, convolve3), dim=2)
        return output

class Middle(nn.Module):
    def __init__(self, p_dropout=0.5, length=60):
        super(Middle, self).__init__()
        self.activation = nn.ReLU()
        self.p_dropout = p_dropout
        self.softmax = nn.Softmax(dim=1)
        # Set the device to either CUDA or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transformer
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.pos_encoder = PositionalEncoding(ENV_HIDDEN_SIZE, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(ENV_HIDDEN_SIZE, nhead=2, batch_first=True, dim_feedforward=2048, activation="relu", dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers, enable_nested_tensor=False)

        # Generate the mask and move it to the correct device
        self.transformer_mask = generate_square_subsequent_mask(length).to(self.device)

    def forward(self, fromencoder, input_masking, training=True):
        src = fromencoder * math.sqrt(ENV_HIDDEN_SIZE)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=input_masking)  # outputs probably
        return output

class Decoder(nn.Module):
    def __init__(self, slot_size, intent_size, dropout_p=0.5, LENGTH=60):
        super(Decoder, self).__init__()
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.dropout_p = dropout_p
        self.softmax = nn.Softmax(dim=1)

        # Set the device to either CUDA or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, ENV_HIDDEN_SIZE)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.slot_trans = nn.Linear(ENV_HIDDEN_SIZE, self.slot_size)
        self.intent_out = nn.Linear(ENV_HIDDEN_SIZE, self.intent_size)
        self.intent_out_cls = nn.Linear(ENV_EMBEDDING_SIZE, self.intent_size)  # dim of bert
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=ENV_HIDDEN_SIZE, nhead=2, batch_first=True, dim_feedforward=300, activation="relu")
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        # Generate the masks and move them to the correct device
        self.transformer_mask = generate_square_subsequent_mask(LENGTH).to(self.device)
        self.transformer_diagonal_mask = generate_square_diagonal_mask(LENGTH).to(self.device)

        self.pos_encoder = PositionalEncoding(ENV_HIDDEN_SIZE, dropout=0.1)
        self.self_attention = nn.MultiheadAttention(embed_dim=ENV_HIDDEN_SIZE, num_heads=8, dropout=0.1, batch_first=True)
        self.layer_norm = nn.LayerNorm(ENV_HIDDEN_SIZE)

    def forward(self, input, encoder_outputs, encoder_maskings, tag2index, bert_subtoken_maskings=None, infer=False):
        # encoder outputs: BATCH, LENGTH, Dims (16,60,1024)
        batch_size = encoder_outputs.shape[0]
        length = encoder_outputs.size(1)  # for every token in batches
        embedded = self.embedding(input)

        # Self-attention
        encoder_outputs2 = encoder_outputs
        context, attn_weight = self.self_attention(encoder_outputs2, encoder_outputs2, encoder_outputs2, key_padding_mask=encoder_maskings)
        encoder_outputs2 = self.layer_norm(self.dropout2(context)) + encoder_outputs2
        sum_mask = (~encoder_maskings).sum(1).unsqueeze(1)
        sum_encoder = ((encoder_outputs2) * ((~encoder_maskings).unsqueeze(2))).sum(1)
        intent_score = self.intent_out(self.dropout1(sum_encoder / sum_mask))  # B, D

        newtensor = torch.zeros(batch_size, length, ENV_HIDDEN_SIZE, device=self.device)
        for i in range(batch_size):  # per batch
            newtensor_index = 0
            for j in range(length):  # for each token
                if bert_subtoken_maskings[i][j].item() == 1:
                    newtensor[i][newtensor_index] = encoder_outputs[i][j]
                    newtensor_index += 1

        if infer == False:
            embedded = embedded * math.sqrt(ENV_HIDDEN_SIZE)
            embedded = self.pos_encoder(embedded)
            zol = self.transformer_decoder(
                tgt=embedded,
                memory=newtensor,
                memory_mask=self.transformer_diagonal_mask,
                tgt_mask=self.transformer_mask
            )

            scores = self.slot_trans(self.dropout3(zol))
            slot_scores = F.log_softmax(scores, dim=2)
        else:
            bos = Variable(torch.LongTensor([[tag2index['<BOS>']] * batch_size])).to(self.device).transpose(1, 0)
            bos = self.embedding(bos)
            tokens = bos
            for i in range(length):
                temp_embedded = tokens * math.sqrt(ENV_HIDDEN_SIZE)
                temp_embedded = self.pos_encoder(temp_embedded)
                zol = self.transformer_decoder(
                    tgt=temp_embedded,
                    memory=newtensor,
                    tgt_mask=self.transformer_mask[:i+1, :i+1],
                    memory_mask=self.transformer_diagonal_mask[:i+1, :]
                )
                scores = self.slot_trans(self.dropout3(zol))
                softmaxed = F.log_softmax(scores, dim=2)
                # Append the last token to vectors
                _, input = torch.max(softmaxed, 2)
                newtok = self.embedding(input)
                tokens = torch.cat((bos, newtok), dim=1)
            slot_scores = softmaxed

        return slot_scores.view(input.size(0) * length, -1), intent_score


class CNet(nn.Module):
    def __init__(self, model_path=None, bert_addr='./bert_models/bert-large-uncased-temp/', padded_length=60):
        super(CNet, self).__init__()
        self.length = padded_length
        self.bert_addr = bert_addr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ✅ Load tokenizer only once
        try:
            self.tokenizer = BertTokenizer.from_pretrained(bert_addr)
        except Exception:
            # fallback if you're using torch.hub
            self.tokenizer = torch.hub.load(bert_addr, 'tokenizer', bert_addr, verbose=False, source="local")

        if model_path:
            self.word2index = load_mapping(f'models/ctran{_fn}-word2index.pkl')
            self.index2word = load_mapping(f'models/ctran{_fn}-index2word.pkl')
            self.tag2index = load_mapping(f'models/ctran{_fn}-tag2index.pkl')
            self.index2tag = load_mapping(f'models/ctran{_fn}-index2tag.pkl')
            self.intent2index = load_mapping(f'models/ctran{_fn}-intent2index.pkl')
            self.index2intent = load_mapping(f'models/ctran{_fn}-index2intent.pkl')
        else:
            self.word2index, self.index2word, self.tag2index, self.index2tag, self.intent2index, self.index2intent = word2index, index2word, tag2index, index2tag, intent2index, index2intent

        self.bert_layer = BertLayer(bert_addr)
        self.encoder = Encoder(len(self.word2index))
        self.middle = Middle(length=padded_length)
        self.decoder = Decoder(len(self.tag2index), len(self.intent2index), LENGTH=padded_length)

        if model_path:
            # Use map_location to ensure models are loaded onto the right device (CPU if CUDA isn't available)
            # self.bert_layer.load_state_dict(torch.load(f'{model_path}-bertlayer.pkl', map_location=self.device).state_dict())
            self.encoder.load_state_dict(torch.load(f'{model_path}-encoder.pkl', map_location=self.device, weights_only=False).state_dict())
            self.middle.load_state_dict(torch.load(f'{model_path}-middle.pkl', map_location=self.device, weights_only=False).state_dict())
            self.decoder.load_state_dict(torch.load(f'{model_path}-decoder.pkl', map_location=self.device, weights_only=False).state_dict())

        if torch.cuda.is_available():
            self.bert_layer = self.bert_layer.cuda()
            self.encoder = self.encoder.cuda()
            self.middle = self.middle.cuda()
            self.decoder = self.decoder.cuda()


    def forward(self, bert_info, input, encoder_maskings, bert_subtoken_maskings=None, infer=False):
        # Process the input through the BERT layer
        bert_last_hidden, bert_pooler_output = self.bert_layer(bert_info)

        # Pass the BERT last hidden state through the encoder
        encoder_output = self.encoder(bert_last_hidden)

        # Pass the encoder output through the middle component
        middle_output = self.middle(encoder_output, encoder_maskings)

        # Pass the middle output and other inputs through the decoder
        slot_scores, intent_score = self.decoder(input, middle_output, encoder_maskings, self.tag2index,bert_subtoken_maskings, infer)

        return slot_scores, intent_score


class CNetOnnx(nn.Module):
    def __init__(self, model_path=None, bert_onnx_path=None, bert_addr='./bert_models/bert-large-uncased-temp/', padded_length=60):
        super(CNetOnnx, self).__init__()
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

        # Replace PyTorch-based BertLayer with ONNX-based one
        self.bert_layer = BertLayerONNX(bert_onnx_path)
        self.encoder = Encoder(len(self.word2index))
        self.middle = Middle(length=padded_length)
        self.decoder = Decoder(len(self.tag2index), len(self.intent2index), LENGTH=padded_length)

        if model_path:
            # self.bert_layer.load_state_dict(torch.load(f'{model_path}-bertlayer.pkl', map_location=self.device).state_dict())
            self.encoder.load_state_dict(torch.load(f'{model_path}-encoder.pkl', map_location=self.device, weights_only=False).state_dict())
            self.middle.load_state_dict(torch.load(f'{model_path}-middle.pkl', map_location=self.device, weights_only=False).state_dict())
            self.decoder.load_state_dict(torch.load(f'{model_path}-decoder.pkl', map_location=self.device, weights_only=False).state_dict())

        if torch.cuda.is_available():
            # self.bert_layer = self.bert_layer.cuda()
            self.encoder = self.encoder.cuda()
            self.middle = self.middle.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, bert_info, input, encoder_maskings, bert_subtoken_maskings=None, infer=False):
        # Process the input through the ONNX-based BERT layer
        bert_tokens, bert_mask, bert_tok_typeid = bert_info
        bert_last_hidden, bert_pooler_output = self.bert_layer.forward(bert_tokens, bert_mask, bert_tok_typeid)

        # Pass the BERT last hidden state through the encoder
        encoder_output = self.encoder(bert_last_hidden)

        # Pass the encoder output through the middle component
        middle_output = self.middle(encoder_output, encoder_maskings)

        # Pass the middle output and other inputs through the decoder
        slot_scores, intent_score = self.decoder(input, middle_output, encoder_maskings, self.tag2index, bert_subtoken_maskings, infer)

        return slot_scores, intent_score

class BertLayerTRT:
    def __init__(self, trt_engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        # Load TensorRT Engine
        with open(trt_engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        # print(f"Execution Context: {self.context}")
        # print(f"Available Providers: {self.engine}")
        # print(f"Number of Optimization Profiles: {self.engine.num_optimization_profiles}")

        # Use `num_io_tensors` instead of `num_bindings`
        # print(f"Number of I/O Tensors (Bindings): {self.engine.num_io_tensors}")

        # Print Memory Info (Updated to avoid deprecated methods)
        # print(f"Device Memory (bytes): {self.engine.device_memory_size}")

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
        batch_size, seq_len, hidden = encoder_outputs.shape

        # Set input shapes dynamically
        self.context.set_input_shape("encoder_outputs", (batch_size, seq_len, hidden))
        self.context.set_input_shape("encoder_maskings", (batch_size, seq_len))

        # Copy input data
        cuda.memcpy_htod(self.device_buffers["encoder_outputs"], encoder_outputs.cpu().numpy().astype(self.dtypes["encoder_outputs"]))
        cuda.memcpy_htod(self.device_buffers["encoder_maskings"], encoder_maskings.cpu().numpy().astype(self.dtypes["encoder_maskings"]))

        # Run inference
        self.context.execute_v2(self.bindings)

        # Retrieve outputs
        slot_scores_np = np.empty(self.shapes["slot_scores"], dtype=self.dtypes["slot_scores"])
        intent_score_np = np.empty(self.shapes["intent_score"], dtype=self.dtypes["intent_score"])

        cuda.memcpy_dtoh(slot_scores_np, self.device_buffers["slot_scores"])
        cuda.memcpy_dtoh(intent_score_np, self.device_buffers["intent_score"])

        return torch.tensor(slot_scores_np, device=self.device), torch.tensor(intent_score_np, device=self.device)

class DecoderOnnx(nn.Module):
    def __init__(self, onnx_path, device="cuda"):
        super().__init__()
        self.session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def forward(self, input_tensor, encoder_outputs, encoder_maskings, bert_subtoken_maskings=None):
        # Only use encoder_outputs and encoder_maskings because ONNX expects only those
        inputs = {
            "encoder_outputs": encoder_outputs.cpu().numpy(),
            "encoder_maskings": encoder_maskings.cpu().numpy()
        }

        # Run inference
        outputs = self.session.run(["slot_scores", "intent_score"], inputs)

        # Convert outputs back to tensors on the correct device
        slot_scores = torch.tensor(outputs[0], device=self.device)
        intent_score = torch.tensor(outputs[1], device=self.device)
        return slot_scores, intent_score


class DecoderInfer(nn.Module):
    def __init__(self, full_decoder):
        super().__init__()
        self.full_decoder = full_decoder

    def forward(self, input, encoder_outputs, encoder_maskings, bert_subtoken_maskings):
        return self.full_decoder.forward(input, encoder_outputs, encoder_maskings, bert_subtoken_maskings, infer=True)


class CNetTRT(nn.Module):
    def __init__(self, model_path=None, bert_trt_path=None, decoder_trt_path='/data/EEtest/decoder_infer.trt', bert_addr='./bert_models/bert-large-uncased-temp/', padded_length=60):
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
        self.encoder = Encoder(len(self.word2index))
        self.middle = Middle(length=padded_length)
        # self.decoder = Decoder(len(self.tag2index), len(self.intent2index), LENGTH=padded_length)
        self.decoder = DecoderTRT(decoder_trt_path)
        if model_path:
            import __main__
            # __main__.BertLayer = BertLayer  # Ensure BertLayer is available in the namespace
            __main__.Encoder = Encoder  # Ensure Encoder is available in the namespace
            __main__.Middle = Middle  # Ensure Middle is available in the namespace
            # __main__.Decoder = Decoder  # Ensure Decoder is available in the namespace
            __main__.PositionalEncoding = PositionalEncoding  # Ensure PositionalEncoding is available in the namespace
            
            # self.bert_layer.load_state_dict(torch.load(f'{model_path}-bertlayer.pkl', map_location=self.device).state_dict())
            self.encoder.load_state_dict(torch.load(f'{model_path}-encoder.pkl', map_location=self.device, weights_only=False).state_dict())
            self.middle.load_state_dict(torch.load(f'{model_path}-middle.pkl', map_location=self.device, weights_only=False).state_dict())
            # self.decoder.load_state_dict(torch.load(f'{model_path}-decoder.pkl', map_location=self.device, weights_only=False).state_dict())


        # if model_path:
        #     self.encoder.load_state_dict(torch.load(f'{model_path}-encoder.pkl', map_location=self.device, weights_only=False).state_dict())
        #     self.middle.load_state_dict(torch.load(f'{model_path}-middle.pkl', map_location=self.device, weights_only=False).state_dict())
        #     self.decoder.load_state_dict(torch.load(f'{model_path}-decoder.pkl', map_location=self.device, weights_only=False).state_dict())

        # if torch.cuda.is_available():
        #     self.encoder = self.encoder.cuda()
        #     self.middle = self.middle.cuda()
        #     self.decoder = self.decoder.cuda()

    def forward(self, bert_info, input, encoder_maskings, bert_subtoken_maskings=None, infer=False):
        # Process the input through the TensorRT-based BERT layer
        bert_tokens, bert_mask, bert_tok_typeid = bert_info
        bert_last_hidden, bert_pooler_output = self.bert_layer.forward(bert_tokens, bert_mask, bert_tok_typeid)
        # Pass the BERT last hidden state through the encoder
        encoder_output = self.encoder(bert_last_hidden)
        # Pass the encoder output through the middle component
        middle_output = self.middle(encoder_output, encoder_maskings)
        # Pass the middle output and other inputs through the decoder
        # slot_scores, intent_score = self.decoder(input, middle_output, encoder_maskings, self.tag2index, bert_subtoken_maskings, infer)
        slot_scores, intent_score = self.decoder(input, middle_output, encoder_maskings, bert_subtoken_maskings)
        return slot_scores, intent_score
