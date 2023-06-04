# Константные индексы особых токенов
BOS_IDX = 0 # begin of sentence
EOS_IDX = 1 # end of sentence
PAD_IDX = 2 # index for fulfilling
UNK_IDX = 3 # index for unknown token

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 128
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

Bad_list_word = ["бля", "жоп", "еба", "ху", "пидо", "пизд", 
                 "ганд", "долб", "сц", "дроч", "порн", "сук",
                 "муд", "задр", "инце", "звон", "пень", "залуп", "гавн", 
                 "говн", "ссат", "срат", "сос", "перн", "чмо", "проститу",
                 "шалав", "камш", "вебка", "презерв", "дрист", "зае", "сись",
                 "оргаз", "в рот", "сел", "смс", "регистрации", "взро", "чува", "муж", 
                 "жен", "сц", "срет", "видео", "видос", "срет"]

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

flags = {"smalltalk_greetings" : 0,
         "smalltalk_bye" : 0,
         "book_room" : 0,
         "get_room" : 0,
         "common_qstn" : 0
         }

ideal_flags = {"smalltalk_greetings" : 2,
         "smalltalk_bye" : 0,
         "book_room" : 0,
         "get_room" : 0,
         "common_qstn" : 0
         }
