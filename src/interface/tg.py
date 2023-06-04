from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

import sys
import os
import csv
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../registrator')))

import registrator.classification as cls
import registrator.generation as gn
import registrator.params as p

f = open("log.txt", "a")
f.write("===================================")

def make_log(log_date, log_time, log_exp_time, role, replic, id):
    f = open(f"log_txt_{id}.txt", "a")
    log = log_date + " | " + log_time + " | " + log_exp_time + " | "\
        + role + " | " + replic + "\n"
    f.write(log)
    f.close()

def make_csv_log(log_date, log_time, log_exp_time, role, replic, id):
    f = open(f"log_csv_{id}.csv", "a", newline = '') 
    fieldnames = ['log_date', 'log_time', 'log_exp_time', 'role', 'replic']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow({'log_date' : log_date, 'log_time' : log_time, 
                     'log_exp_time' : log_exp_time, 'role' : role, 
                     'replic' : replic})
    f.close()

def get_logs(start_time):
    cur_time = datetime.now()
    log_date, log_time = (str(cur_time)).split(" ")
    log_exp_time = str(cur_time - start_time)
    return log_date, log_time, log_exp_time

hotel_registrator = Bot(token = "5641053269:AAEGV58MDv_uGvFOv1vtExJC8FP20X5aguU")
dp = Dispatcher(hotel_registrator)

def post_process(replic):
    text = str(replic)
    text = text.lstrip().rstrip().capitalize()
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    return text

def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k

sample_category = ["smalltalk_greetings", "get_room", "book_room", "get_room", "smalltalk_bye"]

if __name__ == "__main__":
    classificator = cls.Classificator()
    gen = gn.Generator()

    prev_category = -1
    prev_answer_dict = {}
    prev_replic_dict = {}
    start_time_dict = {}
    prev_answer = ""
    prev_replic = ""
    idx = 0
    
    @dp.message_handler()
    async def msg_send(message : types.Message):
        f = open(f"log{message.from_id}.txt", "a")
        
        global prev_category
        global prev_answer
        global prev_answer_dict
        global prev_replic_dict
        global prev_replic
        global strat_time_dict
        global idx
        
        if(message.text == "start"):
            f = open(f"log_csv_{message.from_id}.csv", "a", newline = '') 
            fieldnames = ['log_date', 'log_time', 'log_exp_time', 'role', 'replic']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            f.close()
            start_time_dict[message.from_id] = datetime.now()
            return
        
        if(message.text == "exit"):
            prev_category = -1
            prev_answer = ""
            prev_answer_dict[message.from_id] = ""
            prev_replic = ""
            prev_replic_dict[message.from_id] = ""
            idx = 0
            f.write("====================" + '\n')
            f.close()
            return
        
        log_date, log_time, log_exp_time = get_logs(start_time_dict[message.from_id])
        make_log(log_date, log_time, log_exp_time, "R", str(message.text), message.from_id)
        make_csv_log(log_date, log_time, log_exp_time, "R", str(message.text), message.from_id)
        
        f.write(message.text + '\n')
        if (start_time_dict.get(message.from_id) == None):
            start_time_dict[message.from_id] = datetime.now()
        
        if (prev_replic_dict.get(message.from_id) == None):
            prev_replic_dict[message.from_id] = ""
            
        if (prev_answer_dict.get(message.from_id) == None):
            prev_answer_dict[message.from_id] = ""
            
        if(message.text == prev_replic_dict[message.from_id]):
            answer = "Вы уже такое спрашивали\n" + prev_answer_dict[message.from_id]
            await message.answer(answer)
            
            log_date, log_time, log_exp_time = get_logs(start_time_dict[message.from_id])
            make_log(log_date, log_time, log_exp_time, "R", str(answer), message.from_id)
            make_csv_log(log_date, log_time, log_exp_time, "R", str(answer), message.from_id)
            
            f.write("Вы уже такое спрашивали\n" + prev_answer_dict[message.from_id] + '\n') 
            f.close() 
            return   

        prev_replic = message.text
        prev_replic_dict[message.from_id] = message.text
        category = classificator.classify(message.text)
        key = get_key(p.categories_dict, category)

        answer = gen.generate(message.text, category)
        answer = post_process(answer)
        
        log_date, log_time, log_exp_time = get_logs(start_time_dict[message.from_id])
        make_log(log_date, log_time, log_exp_time, "R", str(answer), message.from_id)
        make_csv_log(log_date, log_time, log_exp_time, "R", str(answer), message.from_id)
        
        f.write(answer + '\n')
        prev_answer = answer
        prev_answer_dict[message.from_id] = answer
        f.close()
        await message.answer(answer) 

        
    executor.start_polling(dp, skip_updates=True)