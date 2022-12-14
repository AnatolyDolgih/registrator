import torch
import torch.nn as nn
import pandas as pd

import my_model as tr_model
import data_preparation as dp

from progress.bar import ChargingBar 
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from timeit import default_timer as timer
from torch.nn.utils.rnn import pad_sequence

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, save, output_file

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "D:/SP/Data/"
file_name = "loss_visualization.html"

#####################################
#####     Подготовка данных     #####
#####################################


BOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
UNK_IDX = 3

train_data = dp.readTrainData(data_path)
valid_data = dp.readValidData(data_path)

vocab_src, vocab_trg = dp.load_vocab(dp.spacy_ru, train_data, valid_data)

src_vocab_size = len(vocab_src.vocab)
trg_vocab_size = len(vocab_trg.vocab)    

train_iter = dp.MyDataset(train_data)
valid_iter = dp.MyDataset(valid_data)

text_transform_src = dp.sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               vocab_src, dp.tensor_transform)

text_transform_trg = dp.sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               vocab_trg, dp.tensor_transform)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform_src(src_sample.rstrip("\n")))
        tgt_batch.append(text_transform_trg(tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

################################################
############     Нейронная сеть    #############
################################################


SRC_VOCAB_SIZE = len(vocab_src)
TGT_VOCAB_SIZE = len(vocab_trg)
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 128
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

NUM_EPOCHS = 10

transformer = tr_model.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

def save_nn_in_epoch(model, epoch):
    name = "Models/model_" + str(epoch) + "_epoch.pth"
    name_param = "Models/model_params_" + str(epoch) + "_epoch.pth"
    torch.save(model, name)
    torch.save(model.state_dict(), name_param)


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    bar = ChargingBar("Тренировка ...", max = len(train_dataloader))
    
    for src, tgt in train_dataloader:
        bar.next()
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = tr_model.create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    bar.finish()
    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    valid_dataloader = DataLoader(valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    bar = ChargingBar("Валидация...", max = len(valid_dataloader))
    for src, tgt in valid_dataloader:
        bar.next()
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = tr_model.create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    bar.finish()
    return losses / len(valid_dataloader)

def train():
    epochs = []
    train_l = []
    valid_l = []
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        valid_loss = evaluate(transformer)
        save_nn_in_epoch(transformer, epoch)
        
        epochs.append(epoch)
        train_l.append(train_loss)
        valid_l.append(valid_loss)
    
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {valid_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    df = pd.DataFrame({
        "Epochs": epochs,
        "Train_loss": train_l,
        "Valid_loss": valid_l,
    })
    source = ColumnDataSource(df)
    plot = figure(title = "Зависимость ошибки на тренировочной и валидационной выборках от эпохи",
                  max_width=750, height=500,
                  x_axis_label = "Эпоха",
                  y_axis_label = "Ошибка")
    plot.line(x= 'Epochs', y='Train_loss',
              color='green', alpha=0.8, legend="Train loss", line_width=2,
              source=source)
    plot.line(x= 'Epochs', y='Valid_loss',
              color='red', alpha=0.8, legend='Valid loss', line_width=2,
              source=source)
    output_file(filename = file_name)
    save(plot)
    print("Нейронная сеть обучилась")



torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
train()