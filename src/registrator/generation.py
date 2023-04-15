# Прописываем путь к моделям  
import os
pathToModels = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/'))

import torch
import re
import spacy

from os.path import exists
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from typing import List

from torchtext.data.utils import get_tokenizer

# Константные индексы особых токенов
BOS_IDX = 0 # begin of sentence
EOS_IDX = 1 # end of sentence
PAD_IDX = 2 # index for fulfilling
UNK_IDX = 3 # index for unknown token

class MyDataset(Dataset):
    def __init__(self, list):
        self.data = list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def load_tokenizers():
    spacy_ru = spacy.load("ru_core_news_sm")
    return spacy_ru

spacy_ru = load_tokenizers()
    
def normString(str):
    str = str.lower().rstrip().lstrip()
    str = re.sub(r"([.!?])", r" \1", str)
    str = re.sub(r"[^а-яА-Я.!?]+", r" ", str)
    return str

def readAndSplitData(data_path, proportion):
    lines = []
    with open(f"{data_path}", "r", encoding='cp1251') as f:
        for line in f:
            lines.append(line)
        
    data = [[normString(lines[i]), normString(lines[i+1])] for i in range(0, len(lines), 2)]
    new_data = [data[i] for i in range(0, 10001)]
    train_data = [new_data[i] for i in range(0, round(len(new_data) * proportion))]
    valid_data = [new_data[i] for i in range(round(len(new_data) * proportion), len(new_data))]
    return train_data, valid_data

def tokenize_ru(text):
        return tokenize(text, spacy_ru)
    
def build_vocabulary(spacy_ru, train_data,
                     valid_data):

    print("Составление словаря посетителя ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train_data + valid_data, tokenize_ru, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    
    print("Составление словаря регистратора ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train_data + valid_data, tokenize_ru, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

def load_vocab(spacy_ru, train_data, valid_data, vocab_path):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_ru, train_data, valid_data)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Составление закончено.\nРазмеры словарей:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def get_vocab_size(path):
    vocab_src, vocab_tgt = torch.load(path)
    return len(vocab_src), len(vocab_tgt)

def joinRegistrationDialogs(data_path):
    with open(f"{data_path}", "w") as data_file:
        content = []
        count = []
        for i in range(1, 10001):
            with open(f"{data_path}log_appraisals{i}.txt", "r", encoding='cp1251') as source_file:
                content += source_file.readlines()
            content.pop()
            count.append(len(content))
            
        for i in content:
            print(i.split(':')[0], end='\n', file = data_file)
    return 


# модель заселения в отель
import torch
import torch.nn as nn
import math

from torch import Tensor
from torch.nn import Transformer

# Константные индексы особых токенов
BOS_IDX = 0 # begin of sentence
EOS_IDX = 1 # end of sentence
PAD_IDX = 2 # index for fulfilling
UNK_IDX = 3 # index for unknown token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory, tgt_mask)
        
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# модель ответа на свободный диалог

def load_vocab1(vocab_path):
    vocab_src, vocab_tgt = torch.load(vocab_path)
    return vocab_src, vocab_tgt
  
class Registrator:
    def __init__(self, vocab_path, model_path):
        EMB_SIZE = 256
        NHEAD = 8
        FFN_HID_DIM = 128
        BATCH_SIZE = 256
        NUM_ENCODER_LAYERS = 6
        NUM_DECODER_LAYERS = 6
        self.vocab_src, self.vocab_trg = load_vocab1(vocab_path)
        SRC_VOCAB_SIZE = len(self.vocab_src)
        TGT_VOCAB_SIZE = len(self.vocab_trg)
        self.model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
        self.text_transform_src = sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               self.vocab_src, tensor_transform)

        self.text_transform_trg = sequential_transforms(get_tokenizer('spacy', language='ru_core_news_sm'), 
                                               self.vocab_trg, tensor_transform)        
        self.model.load_state_dict(torch.load(model_path, map_location = device))
        self.model.eval()
        self.model.to(device)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Общее число обучаемых параметров: {pytorch_total_params}")
        
        
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_len-1):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(device)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys

    def generateRegistratorAnswer(self, src_sentence):
        
        self.model.eval()
        src = self.text_transform_src(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=128, start_symbol = BOS_IDX).flatten()
        return " ".join(self.vocab_trg.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<s>", "").replace("</s>", "")        