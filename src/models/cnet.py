import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import pickle

ENV_BERT_ID_CLS=False # use cls token for id classification
ENV_EMBEDDING_SIZE=1024 # dimention of embbeding, bertbase=768,bertlarge&elmo=1024
ENV_BERT_ADDR='./bert-large-uncased/'
ENV_SEED=1331
ENV_CNN_FILTERS=128
ENV_CNN_KERNELS=4
ENV_HIDDEN_SIZE=ENV_CNN_FILTERS*ENV_CNN_KERNELS



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


#start of the shared encoder
class BertLayer(nn.Module):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.bert_model = torch.hub.load(ENV_BERT_ADDR, 'model', ENV_BERT_ADDR,source="local")

    def forward(self, bert_info=None):
        (bert_tokens, bert_mask, bert_tok_typeid) = bert_info
        bert_encodings = self.bert_model(bert_tokens, bert_mask, bert_tok_typeid)
        bert_last_hidden = bert_encodings['last_hidden_state']
        bert_pooler_output = bert_encodings['pooler_output']
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

#Middle
class Middle(nn.Module):
    def __init__(self ,p_dropout=0.5):
        super(Middle, self).__init__()
        self.activation = nn.ReLU()
        self.p_dropout = p_dropout
        self.softmax = nn.Softmax(dim=1)
        #Transformer
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.pos_encoder = PositionalEncoding(ENV_HIDDEN_SIZE, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(ENV_HIDDEN_SIZE, nhead=2,batch_first=True, dim_feedforward=2048 ,activation="relu", dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers,enable_nested_tensor=False)
        self.transformer_mask = generate_square_subsequent_mask(LENGTH).cuda()

    def forward(self, fromencoder,input_masking,training=True):
        src = fromencoder * math.sqrt(ENV_HIDDEN_SIZE)
        src = self.pos_encoder(src)
        output = (self.transformer_encoder(src,src_key_padding_mask=input_masking)) # outputs probably
        return output
    
#start of the decoder
class Decoder(nn.Module):

    def __init__(self,slot_size,intent_size,dropout_p=0.5):
        super(Decoder, self).__init__()
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.dropout_p = dropout_p
        self.softmax= nn.Softmax(dim=1)
        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, ENV_HIDDEN_SIZE)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.slot_trans = nn.Linear(ENV_HIDDEN_SIZE, self.slot_size)
        self.intent_out = nn.Linear(ENV_HIDDEN_SIZE,self.intent_size)
        self.intent_out_cls = nn.Linear(ENV_EMBEDDING_SIZE,self.intent_size) # dim of bert
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=ENV_HIDDEN_SIZE, nhead=2,batch_first=True,dim_feedforward=300 ,activation="relu")
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.transformer_mask = generate_square_subsequent_mask(LENGTH).cuda()
        self.transformer_diagonal_mask = generate_square_diagonal_mask(LENGTH).cuda()
        self.pos_encoder = PositionalEncoding(ENV_HIDDEN_SIZE, dropout=0.1)
        self.self_attention = nn.MultiheadAttention(embed_dim=ENV_HIDDEN_SIZE
                                                    ,num_heads=8,dropout=0.1
                                                    ,batch_first=True)
        self.layer_norm = nn.LayerNorm(ENV_HIDDEN_SIZE)


    def forward(self, input,encoder_outputs,encoder_maskings,tag2index,bert_subtoken_maskings=None,infer=False):
        # encoder outputs: BATCH,LENGTH,Dims (16,60,1024)
        batch_size = encoder_outputs.shape[0]
        length = encoder_outputs.size(1) #for every token in batches
        embedded = self.embedding(input)

        #print("NOT CLS")
        encoder_outputs2=encoder_outputs
        context,attn_weight = self.self_attention(encoder_outputs2,encoder_outputs2,encoder_outputs2
                                                  ,key_padding_mask=encoder_maskings)
        encoder_outputs2 = self.layer_norm(self.dropout2(context))+encoder_outputs2
        sum_mask = (~encoder_maskings).sum(1).unsqueeze(1)
        sum_encoder = ((((encoder_outputs2)))*((~encoder_maskings).unsqueeze(2))).sum(1)
        intent_score = self.intent_out(self.dropout1(sum_encoder/sum_mask)) # B,D


        newtensor = torch.cuda.FloatTensor(batch_size, length,ENV_HIDDEN_SIZE).fill_(0.) # size of newtensor same as original
        for i in range(batch_size): # per batch
            newtensor_index=0
            for j in range(length): # for each token
                if bert_subtoken_maskings[i][j].item()==1:
                    newtensor[i][newtensor_index] = encoder_outputs[i][j]
                    newtensor_index+=1

        if infer==False:
            embedded=embedded*math.sqrt(ENV_HIDDEN_SIZE)
            embedded = self.pos_encoder(embedded)
            zol = self.transformer_decoder(tgt=embedded,memory=newtensor
                                           ,memory_mask=self.transformer_diagonal_mask
                                           ,tgt_mask=self.transformer_mask)

            scores = self.slot_trans(self.dropout3(zol))
            slot_scores = F.log_softmax(scores,dim=2)
        else:
            bos = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
            bos = self.embedding(bos)
            tokens=bos
            for i in range(length):
                temp_embedded=tokens*math.sqrt(ENV_HIDDEN_SIZE)
                temp_embedded = self.pos_encoder(temp_embedded)
                zol = self.transformer_decoder(tgt=temp_embedded,
                                               memory=newtensor,
                                               tgt_mask=self.transformer_mask[:i+1,:i+1],
                                               memory_mask=self.transformer_diagonal_mask[:i+1,:]
                                               )
                scores = self.slot_trans(self.dropout3(zol))
                softmaxed = F.log_softmax(scores,dim=2)
                #the last token is apended to vectors
                _,input = torch.max(softmaxed,2)
                newtok = self.embedding(input)
                tokens=torch.cat((bos,newtok),dim=1)
            slot_scores = softmaxed

        return slot_scores.view(input.size(0)*length,-1), intent_score
    
# CNet class
class CNet(nn.Module):
    def __init__(self, model_path=None):
        super(CNet, self).__init__()
        if model_path:
            self.word2index = load_mapping(f'models/ctran{model_path}-word2index.pkl')
            self.index2word = load_mapping(f'models/ctran{model_path}-index2word.pkl')
            self.tag2index = load_mapping(f'models/ctran{model_path}-tag2index.pkl')
            self.index2tag = load_mapping(f'models/ctran{model_path}-index2tag.pkl')
            self.intent2index = load_mapping(f'models/ctran{model_path}-intent2index.pkl')
            self.index2intent = load_mapping(f'models/ctran{model_path}-index2intent.pkl')

        self.bert_layer = BertLayer()
        self.encoder = Encoder(len(self.word2index))
        self.middle = Middle()
        self.decoder = Decoder(len(self.tag2index), len(self.intent2index))

        if model_path:
            self.bert_layer.load_state_dict(torch.load(f'{model_path}-bertlayer.pkl').state_dict())
            self.encoder.load_state_dict(torch.load(f'{model_path}-encoder.pkl').state_dict())
            self.middle.load_state_dict(torch.load(f'{model_path}-middle.pkl').state_dict())


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