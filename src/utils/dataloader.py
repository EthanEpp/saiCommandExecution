# First step is to import the needed libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score
import math
import re
from torch.utils.data import Dataset, DataLoader
import os


#this function converts tokens to ids and then to a tensor
def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs)).cuda() if torch.cuda.is_available() else Variable(torch.LongTensor(idxs))
    return tensor
# this function turns class text to id
def prepare_intent(intent, to_ix):
    idxs = to_ix[intent] if intent in to_ix.keys() else to_ix["UNKNOWN"]
    return idxs
# converts numbers to <NUM> TAG
def number_to_tag(txt):
    return "<NUM>" if txt.isdecimal() else txt

# Here we remove multiple spaces and punctuation which cause errors in tokenization for bert & elmo.
def remove_punc(mlist):
    mlist = [re.sub(" +"," ",t.split("\t")[0][4:-4]) for t in mlist] # remove spaces down to 1
    temp_train_tokens = []
    # punct remove example:  play samuel-el jackson from 2009 - 2010 > play samuelel jackson from 2009 - 2010
    for row in mlist:
        tokens = row.split(" ")
        newtokens = []
        for token in tokens:
            newtoken = re.sub(r"[.,'\"\\/\-:&’—=–官方杂志¡…“”~%]",r"",token) # remove punc
            newtoken = re.sub(r"[楽園追放�]",r"A",newtoken)
            newtokens.append(newtoken if len(token)>1 else token)
        if newtokens[-1]=="":
            newtokens.pop(-1)
        if newtokens[0]=="":
            newtokens.pop(0)
        temp_train_tokens.append(" ".join(newtokens))
    return temp_train_tokens


# this function returns the main tokens so that we can apply tagging on them. see original paper.
def get_subtoken_mask(current_tokens,bert_tokenizer, length=60):
    temp_mask = []
    for i in current_tokens:
        temp_row_mask = []
        temp_row_mask.append(False) # for cls token
        temp = bert_tokenizer.tokenize(i)
        for j in temp:
            temp_row_mask.append(j[:2]!="##")
        while len(temp_row_mask)<length:
            temp_row_mask.append(False)
        temp_mask.append(temp_row_mask)
        if sum(temp_row_mask)!=len(i.split(" ")):
            print(f"inconsistent:{temp}")
            print(i)
            print(sum(temp_row_mask))
            print(len(i.split(" ")))
    return torch.tensor(temp_mask).cuda()

flatten = lambda l: [number_to_tag(item) for sublist in l for item in sublist]

def tokenize_dataset(dataset_address, bert_addr, length=60):
    # added tokenizer and tokens for
    bert_tokenizer = torch.hub.load(bert_addr, 'tokenizer', bert_addr, verbose=False,source="local")#38toks snips,52Atis
    ##open database and read line by line
    dataset = open(dataset_address,"r").readlines()
    print("example input:"+dataset[100])
    # print("\nexample input:"+dataset[100])
    ##remove last character of lines -\n- in train file
    dataset = [t[:-1] for t in dataset]
    #converts string to array of tokens + array of tags + target intent [array with x=3 and y dynamic]
    dataset_tokens = remove_punc(dataset)
    dataset_subtoken_mask = get_subtoken_mask(dataset_tokens,bert_tokenizer)
    dataset_toks = bert_tokenizer.batch_encode_plus(dataset_tokens,max_length=length,add_special_tokens=True,return_tensors='pt'
                                                  ,return_attention_mask=True , padding='max_length',truncation=True)
    dataset = [[re.sub(" +"," ",t.split("\t")[0]).split(" "),t.split("\t")[1].split(" ")[:-1],t.split("\t")[1].split(" ")[-1]] for t in dataset]
    #removes BOS, EOS from array of tokens and tags
    dataset = [[t[0][1:-1],t[1][1:],t[2]] for t in dataset]
    return dataset, dataset_subtoken_mask,dataset_toks

##
train,train_subtoken_mask,train_toks = tokenize_dataset(ENV_DATASET_TRAIN, ENV_BERT_ADDR)
test, test_subtoken_mask, test_toks = tokenize_dataset(ENV_DATASET_TEST, ENV_BERT_ADDR)
##


##
#convert above array to separate lists
seq_in,seq_out, intent = list(zip(*train))
seq_in_test,seq_out_test, intent_test = list(zip(*test.copy()))
# Create Sets of unique tokens
vocab = set(flatten(seq_in))
slot_tag = set(flatten(seq_out))
intent_tag = set(intent)
##


# adds paddings
sin=[] #padded input tokens
sout=[] # padded output translated tags
sin_test=[] #padded input tokens
sout_test=[] # padded output translated tags
## adds padding inside input tokens
def add_paddings(seq_in,seq_out):
    sin=[]
    sout=[]
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp)<LENGTH:
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
        sin.append(temp)
        # add padding inside output tokens
        temp = seq_out[i]
        if len(temp)<LENGTH:
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
        sout.append(temp)
    return sin,sout

sin,sout=add_paddings(seq_in,seq_out)
sin_test,sout_test=add_paddings(seq_in_test,seq_out_test)

def create_mappings(vocab, slot_tag, intent_tag):
    # making dictionary (token:id), initial value
    word2index = {'<PAD>': 0, '<UNK>':1,'<BOS>':2,'<EOS>':3,'<NUM>':4}
    # add rest of token list to dictionary
    for token in vocab:
        if token not in word2index.keys():
            word2index[token]=len(word2index)
    #make id to token list ( reverse )
    index2word = {v:k for k,v in word2index.items()}

    # initial tag2index dictionary
    tag2index = {'<PAD>' : 0,'<BOS>':2,'<UNK>':1,'<EOS>':3}
    # add rest of tag tokens to list
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)
    # making index to tag
    index2tag = {v:k for k,v in tag2index.items()}

    #initialize intent to index
    intent2index={'UNKNOWN':0}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)
    index2intent = {v:k for k,v in intent2index.items()}
    return word2index, index2word, tag2index, index2tag, intent2index, index2intent

word2index, index2word, tag2index, index2tag, intent2index, index2intent = create_mappings(vocab, slot_tag, intent_tag)

#defining datasets.
def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

class NLUDataset(Dataset):
    def __init__(self, sin,sout,intent,input_ids,attention_mask,token_type_ids,subtoken_mask):
        self.sin = [prepare_sequence(temp,word2index) for temp in sin]
        self.sout = [prepare_sequence(temp,tag2index) for temp in sout]
        self.intent = Variable(torch.LongTensor([prepare_intent(temp,intent2index) for temp in intent])).cuda()
        self.input_ids=input_ids.cuda()
        self.attention_mask=attention_mask.cuda()
        self.token_type_ids=token_type_ids.cuda()
        self.subtoken_mask=subtoken_mask.cuda()
        self.x_mask = [Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, t )))).cuda() for t in self.sin]
    def __len__(self):
        return len(self.intent)
    def __getitem__(self, idx):
        sample = self.sin[idx],self.sout[idx],self.intent[idx],self.input_ids[idx],self.attention_mask[idx],self.token_type_ids[idx],self.subtoken_mask[idx],self.x_mask[idx]
        return sample
#making single list
train_data=NLUDataset(sin,sout,intent,train_toks['input_ids'],train_toks['attention_mask'],train_toks['token_type_ids'],train_subtoken_mask)
test_data=NLUDataset(sin_test,sout_test,intent_test,test_toks['input_ids'],test_toks['attention_mask'],test_toks['token_type_ids'],test_subtoken_mask)
train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# we put all tags inside of the batch in a flat array for F1 measure.
# we use masking so that we only non PAD tokens are counted in f1 measurement
def mask_important_tags(predictions,tags,masks):
    result_tags=[]
    result_preds=[]
    for pred,tag,mask in zip(predictions.tolist(),tags.tolist(),masks.tolist()):
        #index [0] is to get the data
        for p,t,m in zip(pred,tag,mask):
            if not m:
                result_tags.append(p)
                result_preds.append(t)
        #result_tags.pop()
        #result_preds.pop()
    return result_preds,result_tags


def removepads(toks, clip=False):
    global clipindex
    result = toks.copy()
    for i, t in enumerate(toks):
        if t == "<PAD>":
            result.remove(t)
        elif t == "<EOS>":
            result.remove(t)
            if not clip:
                clipindex = i
    if clip:
        result = result[:clipindex]
    return result

def tag_sentence(sentence):
    # Convert to lowercase and replace punctuation with spaces
    sentence = re.sub(r'[^\w\s]', ' ', sentence.lower())

    # Splitting the sentence into tokens
    tokens = re.split(r'(\s+)', sentence)

    # Initialize variables
    output_tokens = []
    output_labels = []

    # Iterate over each token
    for token in tokens:
        if token.strip():  # Avoiding empty tokens
            output_labels.append('O')
            output_tokens.append(token.strip())

    formatted_sentence = ' '.join(output_tokens)
    formatted_labels = ' '.join(output_labels)

    return f"BOS {formatted_sentence} EOS\t O {formatted_labels} open_app"


def tokenize_sample(sample, bert_addr, length=60):
    # length = length
    sample = tag_sentence(sample)
    # added tokenizer and tokens for
    bert_tokenizer = torch.hub.load(bert_addr, 'tokenizer', bert_addr, verbose=False, source="local")
    sample = [sample]
    print("example input:" + sample[0])
    # converts string to array of tokens + array of tags + target intent [array with x=3 and y dynamic]
    sample_tokens = remove_punc(sample)
    sample_subtoken_mask = get_subtoken_mask(sample_tokens, bert_tokenizer, length)
    sample_toks = bert_tokenizer.batch_encode_plus(sample_tokens, max_length=length, add_special_tokens=True, return_tensors='pt',
                                                   return_attention_mask=True, padding='max_length', truncation=True)
    sample = [[re.sub(" +", " ", t.split("\t")[0]).split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in sample]
    # removes BOS, EOS from array of tokens and tags
    sample = [[t[0][1:-1], t[1][1:], t[2]] for t in sample]
    return sample, sample_subtoken_mask, sample_toks

