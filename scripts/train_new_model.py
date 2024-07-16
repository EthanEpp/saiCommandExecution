import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import sys
import os

# Add the parent directory to the Python path so it can find the utils module
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.cnet import CNet


#these are related to training
BATCH_SIZE=4
LENGTH=60
STEP_SIZE=50

# Initialize loss functions
loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
loss_function_2 = nn.CrossEntropyLoss()

# Initialize the model
model = CNet()

# Combine parameters for optimizers
optimizer = optim.AdamW([
    {'params': model.bert_layer.parameters(), 'lr': 0.00001},
    {'params': model.encoder.parameters(), 'lr': 0.001},
    {'params': model.middle.parameters(), 'lr': 0.0001},
    {'params': model.decoder.parameters(), 'lr': 0.0001}
])

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

# Training settings
max_id_prec = 0.
max_sf_f1 = 0.
max_id_prec_both = 0.
max_sf_f1_both = 0.

for step in tqdm(range(STEP_SIZE)):
    losses = []
    id_precision = []
    sf_f1 = []

    ### TRAIN
    model.train()  # set the model to train mode

    for i, (x, tag_target, intent_target, bert_tokens, bert_mask, bert_toktype, subtoken_mask, x_mask) in enumerate(train_data):
        batch_size = tag_target.size(0)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        bert_info = (bert_tokens, bert_mask, bert_toktype)
        start_decode = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
        start_decode = torch.cat((start_decode,tag_target[:,:-1]),dim=1)
        tag_score, intent_score = model(bert_info, start_decode, bert_mask==0, bert_subtoken_maskings=subtoken_mask)

        # Compute losses
        loss_1 = loss_function_1(tag_score, tag_target.view(-1))
        loss_2 = loss_function_2(intent_score, intent_target)
        loss = loss_1 + loss_2
        losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy()[0])
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # Compute metrics
        id_precision.append(accuracy_score(intent_target.detach().cpu(), torch.argmax(intent_score, dim=1).detach().cpu()))
        pred_list, target_list = mask_important_tags(torch.argmax(tag_score, dim=1).view(batch_size, LENGTH), tag_target, x_mask)
        sf_f1.append(f1_score(pred_list, target_list, average="micro", zero_division=0))

    # Print training report
    print("Step", step, "batches", i, ":")
    print("Train-")
    print(f"loss:{round(float(np.mean(losses)), 4)}")
    print(f"SlotFilling F1:{round(float(np.mean(sf_f1)), 3)}")
    print(f"IntentDet Prec:{round(float(np.mean(id_precision)), 3)}")

    losses = []
    sf_f1 = []
    id_precision = []

    #### TEST
    model.eval()  # set the model to eval mode
    with torch.no_grad():  # turn off gradients computation
        for i, (x, tag_target, intent_target, bert_tokens, bert_mask, bert_toktype, subtoken_mask, x_mask) in enumerate(test_data):
            batch_size = tag_target.size(0)

            # Forward pass
            bert_info = (bert_tokens, bert_mask, bert_toktype)
            start_decode = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
            tag_score, intent_score = model(bert_info, start_decode, bert_mask==0, bert_subtoken_maskings=subtoken_mask, infer=True)

            # Compute losses
            loss_1 = loss_function_1(tag_score, tag_target.view(-1))
            loss_2 = loss_function_2(intent_score, intent_target)
            loss = loss_1 + loss_2
            losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy()[0])

            # Compute metrics
            id_precision.append(accuracy_score(intent_target.detach().cpu(), torch.argmax(intent_score, dim=1).detach().cpu()))
            pred_list, target_list = mask_important_tags(torch.argmax(tag_score, dim=1).view(batch_size, LENGTH), tag_target, x_mask)
            sf_f1.append(f1_score(pred_list, target_list, average="micro", zero_division=0))

    # Print testing report
    print("Test-")
    print(f"loss:{round(float(np.mean(losses)), 4)}")
    print(f"SlotFilling F1:{round(float(np.mean(sf_f1)), 4)}")
    print(f"IntentDet Prec:{round(float(np.mean(id_precision)), 4)}")
    print("--------------")

    # Update the best scores
    max_sf_f1 = max(max_sf_f1, round(float(np.mean(sf_f1)), 4))
    max_id_prec = max(max_id_prec, round(float(np.mean(id_precision)), 4))

    if max_sf_f1_both <= round(float(np.mean(sf_f1)), 4) and max_id_prec_both <= round(float(np.mean(id_precision)), 4):
        max_sf_f1_both = round(float(np.mean(sf_f1)), 4)
        max_id_prec_both = round(float(np.mean(id_precision)), 4)
        torch.save(model.bert_layer, f"models/ctran{_fn}-bertlayer.pkl")
        torch.save(model.encoder, f"models/ctran{_fn}-encoder.pkl")
        torch.save(model.middle, f"models/ctran{_fn}-middle.pkl")
        torch.save(model.decoder, f"models/ctran{_fn}-decoder.pkl")
        print("saved")

    # Step the scheduler
    scheduler.step()

print(f"max single SF F1: {max_sf_f1}")
print(f"max single ID PR: {max_id_prec}")
print(f"max mutual SF:{max_sf_f1_both}  PR: {max_id_prec_both}")
