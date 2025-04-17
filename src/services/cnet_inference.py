import re
import torch
import sys
import os
from torch.autograd import Variable
import time

from src.models.cnet import CNet
from src.utils.dataloader import tokenize_sample
def predict_intent_and_tags(sample, sample_toks, sample_subtoken_mask, model, use_cuda=True):
    """
    Predicts the intent and tags for a given sample using a pretrained model.

    Args:
    sample (list): List containing the text split into tokens.
    sample_toks (Tensor): Encoded tokens using the BERT tokenizer.
    sample_subtoken_mask (Tensor): Mask indicating which tokens are 'real' vs subtokens.
    model (nn.Module): Pretrained CNet model.
    use_cuda (bool): Flag to indicate whether to use CUDA (GPU support).

    Returns:
    dict: A dictionary containing the intent and the significant tags.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    word2index = model.word2index
    index2tag = model.index2tag
    index2intent = model.index2intent

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        timings = {}

        start_time = time.time()
        # Move tensors to the appropriate device
        bert_tokens = sample_toks['input_ids'][0].unsqueeze(0).to(device)
        bert_mask = sample_toks['attention_mask'][0].unsqueeze(0).to(device)
        bert_toktype = sample_toks['token_type_ids'][0].unsqueeze(0).to(device)
        subtoken_mask = sample_subtoken_mask[0].unsqueeze(0).to(device)
        timings['bert_input_prep'] = time.time() - start_time

        start_time = time.time()
        start_decode = Variable(torch.LongTensor([[word2index['<BOS>']] * 1]).to(device)).transpose(1, 0)
        timings['sequence_prep'] = time.time() - start_time

        start_time = time.time()
        bert_info = (bert_tokens, bert_mask, bert_toktype)
        tag_score, intent_score = model(bert_info, start_decode, bert_mask == 0, subtoken_mask, infer=True)
        timings['model_processing'] = time.time() - start_time

        # Process tag predictions
        tag_predictions = torch.argmax(tag_score, -1).squeeze().cpu().numpy()
        filtered_tags = [(sample[0][0][i], index2tag[tag]) for i, tag in enumerate(tag_predictions) if index2tag[tag] != 'O' and i < len(sample[0][0])]

        # Process intent prediction
        _, predicted_intent_idx = torch.max(intent_score, -1)
        predicted_intent = index2intent[predicted_intent_idx.item()]

        result = {
            "intent": predicted_intent,
            "tags": filtered_tags,
            "timings": timings
        }
        return result

# def predict_intent_and_tags(sample, sample_toks, sample_subtoken_mask, model, word2index, index2tag, index2intent, use_cuda=True):
#     """
#     Predicts the intent and tags for a given sample using a pretrained model.

#     Args:
#     sample (list): List containing the text split into tokens.
#     sample_toks (Tensor): Encoded tokens using the BERT tokenizer.
#     sample_subtoken_mask (Tensor): Mask indicating which tokens are 'real' vs subtokens.
#     model (nn.Module): Pretrained CNet model.
#     word2index (dict): Dictionary mapping words to their indices.
#     index2tag (dict): Dictionary mapping tag indices to tag labels.
#     index2intent (dict): Dictionary mapping intent indices to intent labels.
#     use_cuda (bool): Flag to indicate whether to use CUDA (GPU support).

#     Returns:
#     dict: A dictionary containing the intent and the significant tags.
#     """
#     # Ensure the model is in evaluation mode
#     model.eval()

#     device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

#     with torch.no_grad():
#         timings = {}

#         start_time = time.time()
#         # Move tensors to the appropriate device
#         bert_tokens = sample_toks['input_ids'][0].unsqueeze(0).to(device)
#         bert_mask = sample_toks['attention_mask'][0].unsqueeze(0).to(device)
#         bert_toktype = sample_toks['token_type_ids'][0].unsqueeze(0).to(device)
#         subtoken_mask = sample_subtoken_mask[0].unsqueeze(0).to(device)
#         timings['bert_input_prep'] = time.time() - start_time

#         start_time = time.time()
#         start_decode = Variable(torch.LongTensor([[word2index['<BOS>']] * 1]).to(device)).transpose(1, 0)
#         timings['sequence_prep'] = time.time() - start_time

#         start_time = time.time()
#         bert_info = (bert_tokens, bert_mask, bert_toktype)
#         tag_score, intent_score = model(bert_info, start_decode, bert_mask == 0, subtoken_mask, infer=True)
#         timings['model_processing'] = time.time() - start_time

#         # Process tag predictions
#         tag_predictions = torch.argmax(tag_score, -1).squeeze().cpu().numpy()
#         filtered_tags = [(sample[0][0][i], index2tag[tag]) for i, tag in enumerate(tag_predictions) if index2tag[tag] != 'O' and i < len(sample[0][0])]

#         # Process intent prediction
#         _, predicted_intent_idx = torch.max(intent_score, -1)
#         predicted_intent = index2intent[predicted_intent_idx.item()]

#         result = {
#             "intent": predicted_intent,
#             "tags": filtered_tags,
#             "timings": timings
#         }
#         return result
    
def run_inference(input_text, model):
    sample, sample_subtoken_mask, sample_toks = tokenize_sample(input_text, model.bert_addr, model.length)
    results = predict_intent_and_tags(sample, sample_toks, sample_subtoken_mask, model)
    return results
