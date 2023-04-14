import re
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification


categories_dict = {"free_dialog": 0, "fio": 1,
                   "smalltalk_greetings": 2, 
                   "smalltalk_bye": 3,
                   "book_room": 4,
                   "get_room": 5,
                   "common_qstn": 6, 
                   "no_category": 7}

reg_word = ["извините", "прощения",
            "добрый веч", "здравствуйте", "приве",
            "нужен номер", "места", "остановиться", "номер",
            "бронь на меня", "бронь на меня", "номер будет", "номер брони",
            "будет забронирован", "бронь на меня", "бронь будет",
            "да, пожалуйста", "не моя вина", "можно, побыстрее",
            "одна ночь", "одну ночь", "номер на одного", "только я",
            "никого нет", "я постою", "хорошо", "побыстрее", "я устал",
            "нет проблем", "спешу", "подешевле", "большой", "маленький",
            "средний", "отличное", "день был", "неплохо", "бронь"
            ]


class Classificator():
    def __init__(self, cls_path):
        print("\033[33m{}".format("Создание классификатора"))
        self.category = categories_dict
        self.model = torch.load(cls_path, map_location=torch.device('cpu'))
        self.tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        print("\033[32m{}\033[0m".format("Модель классификатора загружена"))
        
    def name_classification(self, text):
        """Определяет, является ли text ФИО пользователя"""
        text = str(text)
        text.strip()
        pattern = r'((\b[A-Я][^A-Я\s\.\,][a-я]*)(\s+)([A-Я][a-я]*)'+\
        '(\.+\s*|\s+)([A-Я][a-я]*))'
        name = re.findall(pattern, text)
        if name:
            return True
        else: 
            return False
    
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
        for i in reg_word:
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
        if(self.find_name(text)):
            return self.category["fio"]
        else:
            return self.category["unknown category"]
        
if __name__ == "__main__":
    сl = Classificator()
    