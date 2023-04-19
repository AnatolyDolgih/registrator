import params as p

sample_seq = [p.categories_dict["smalltalk_greeting"], 
              p.categories_dict["get_room"],
              p.categories_dict["book_room"],]

dialog_sequence = ["smalltalk_greeting", "smalltalk_greeting" ]
dialog = []
intents = []  # хранит последовательность интентов пользователя

class DialogControl():
    def __init__(self):
        pass
    
    def process_replic(self, intent, last_intent, old_replic):
        if (intent == p.categories_dict["free_dialog"]):
            return 0
        elif(intent == last_intent):
            return 0
     
    def generate_answer(self, category):
        
        pass
    