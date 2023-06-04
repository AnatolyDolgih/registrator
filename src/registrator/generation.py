# Прописываем путь к моделям  
import os
pathToModels = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/'))

# Модель для генерации диалога о заселении в отель
import math
import re
import spacy
import torch
import torch.nn as nn
import params as p

from torch import Tensor
from torch.nn import Transformer
from os.path import exists
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from typing import List
from torchtext.data.utils import get_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification




def filter(input_text):
    input_text = input_text.lower()
    check = 0
    for word in p.Bad_list_word:
        if (input_text.find(word) >= 0):
            check += 1
        
    if(check == 0):
        return True
    else:
        return False

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

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

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([p.BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([p.EOS_IDX])))

def post_process(replic):
    text = replic.lstrip().rstrip().capitalize()
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    return text

# класс регистратора
class RegBot():
    def __init__(self):
        # self.model = torch.load(f"{pathToModels}/model_20_epoch.pt", 
        #                         map_location = "cpu")
        
        self.vocab_src, self.vocab_tgt = torch.load(f"{pathToModels}/vocab.pt")
        self.model = Seq2SeqTransformer(p.NUM_ENCODER_LAYERS, p.NUM_DECODER_LAYERS, p.EMB_SIZE,
                                 p.NHEAD, len(self.vocab_src), len(self.vocab_tgt), p.FFN_HID_DIM)
        self.model.load_state_dict(torch.load(f"{pathToModels}/model_params_20_epoch.pth",
                                              map_location = "cpu"))
        self.text_transform_src = sequential_transforms(get_tokenizer('spacy', 
                                                        language='ru_core_news_sm'), 
                                                        self.vocab_src, tensor_transform)

        self.text_transform_trg = sequential_transforms(get_tokenizer('spacy', 
                                                        language='ru_core_news_sm'), 
                                                        self.vocab_tgt, tensor_transform)        
        pass

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device="cpu")) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask    
            
    def greedyDecode(self, src, src_mask, max_len, 
                     start_symbol, device = "cpu"):
        
        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_len-1):
            memory = memory.to(device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(device)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == p.EOS_IDX:
                break
        return ys
    
    def generateAnswer(self, replic):
        self.model.eval()
        src = self.text_transform_src(replic).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedyDecode(src, src_mask, max_len=128, start_symbol = p.BOS_IDX).flatten()
        result = " ".join(self.vocab_tgt.lookup_tokens(list(tgt_tokens.cpu().numpy())))\
                                                    .replace("<s>", "").replace("</s>", "")     
        return result 
        
# класс бота    
class FreeBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f"{pathToModels}/RuDialoGPT/model/")
        self.model = AutoModelForCausalLM.from_pretrained(f"{pathToModels}/RuDialoGPT/model/")
        self.flag = True
        self.count = 0

    def get_length_param(self, text):
        tokens_count = len(self.tokenizer.encode(text))
        if tokens_count <= 15:
            len_param = '1'
        elif tokens_count <= 50:
            len_param = '2'
        elif tokens_count <= 256:
            len_param = '3'
        else:
            len_param = '-'
        return len_param

    def generateAnswer(self, input_replic):
        new_user_input_ids = self.tokenizer.encode(f"|0|{self.get_length_param(input_replic)}|" + 
                                                   input_replic + self.tokenizer.eos_token +  "|1|1|", return_tensors="pt")
                
        bot_input_ids = new_user_input_ids
        
        chat_history_ids = self.model.generate(
            bot_input_ids,
            num_return_sequences=1,
            max_length=512,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature = 0.6,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        result = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return result

class ToxicFilter():
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
        self.tokenizer = BertTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
        
    def check(self, replic):
        if not filter(replic):
            return False
        
        batch = self.tokenizer.encode(replic, return_tensors='pt')
        logits = self.model(batch).logits
        predicted_class_id = logits.argmax().item()
        if (self.model.config.id2label[predicted_class_id] == 'toxic'):
            return False
        else:
            return True

class Generator():
    def __init__(self):
        self.registrator_bot = RegBot()
        self.free_bot = FreeBot()
        self.toxic_filter = ToxicFilter()
        print("Генератор создан")
        pass
    
    def generate(self, replic, category):
        if (category == 0):
            check = True
            while (check == True):
                answer = self.free_bot.generateAnswer(replic)
                if self.toxic_filter.check(answer):
                    check = False 
            return answer
        else:
            return self.registrator_bot.generateAnswer(replic)

if __name__ == "__main__":
    gen = Generator()
    replic = "qw"
    while(replic != "exit"):
        print("Введите фразу: ")
        replic = input()
        print(gen.generate(replic))