""" Данный модуль реализует функционал классификатора"""

# Прописываем путь к моделям  
import os
pathToModels = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/'))

import re
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import params as p



class Classificator():
    """Класс классификатора"""
    
    def __init__(self):
        print("\033[33m{}".format("Создание классификатора"))
        self.category = p.categories_dict
        self.model = torch.load(pathToModels + "\intent_catcher.pt", map_location=torch.device('cpu'))
        self.tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        print("\033[32m{}\033[0m".format("Модель классификатора загружена"))
        
    def name_classification(self, text):
        """Определяет, является ли text ФИО пользователя"""
        fio = text.split(" ")
        for l in fio:
            f = open("D://registrator/registrator/data/male_name.txt", "r", encoding = 'utf-8')
            for line in f:
                a = line.strip().lower()
                if (l == a):
                    f.close()
                    return True
            
            f = open("D://registrator/registrator/data/female_name.txt", "r", encoding = 'utf-8')
            for line in f:
                a = line.strip().lower()
                if (l == a):
                    f.close()
                    return True
        return False
        # text = str(text)
        # text.strip()
        # pattern = r'((\b[A-Я][^A-Я\s\.\,][a-я]*)(\s+)([A-Я][a-я]*)'+\
            
        # '(\.+\s*|\s+)([A-Я][a-я]*))'
        # name = re.findall(pattern, text)
        # if name:
        #     return True
        # else: 
        #     return False
    
    def bin_classification(self, text):
        """Определяет, относится ли данная реплика к теме заселения в отель
        или к свободной теме.

        Args:
            text (str): входящая реплика пользователя

        Returns:
            bool: True - реплика о заселении в отель
                  False - реплика на свободную тему
        """
        text = text.lower().rstrip().lstrip()
        count = 0
        for i in p.reg_word:
            if (text.find(i) >= 0):
                count += 1
        if (count > 0):
            return True
        else:
            return False
    
    def intent_classification(self, text):
        model_input = self.tokenizer.encode(text, return_tensors='pt')
        model_output = self.model.bert.config.id2label[self.model(model_input)['logits'].argmax().item()]
        return str(model_output)
      
    def classify(self, text):
        # проверяем фио
        if(self.name_classification(text)):
            return self.category["fio"]
        
        text = text.lower().lstrip().rstrip()
        # проверяем, какой у нас диалог
        if(not self.bin_classification(text)):
            return self.category["free_dialog"]
        
        return self.category[self.intent_classification(text)]
        
if __name__ == "__main__":
    сl = Classificator()
    print(pathToModels + "\intent_catcher.pt")