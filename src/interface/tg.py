from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../registrator')))

import registrator.classification as cls
import registrator.generation as gn
import registrator.params as p

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
    prev_answer = ""
    prev_replic = ""
    idx = 0
    
    @dp.message_handler()
    async def msg_send(message : types.Message):
        global prev_category
        global prev_answer
        global prev_replic
        global idx
        
        if(message.text == prev_replic):
            await message.answer("Вы уже такое спрашивали\n" + prev_answer)  
            return   
        
        prev_replic = message.text
        category = classificator.classify(message.text)
        key = get_key(p.categories_dict, category)
        
        answer = gen.generate(message.text, category)
        answer = post_process(answer)
        prev_answer = answer
        await message.answer(answer) 
        
        # if (prev_category == -1):
        #     # первая фраза
        #     answer = gen.generate(message.text, category)
        #     answer = post_process(answer)
        #     prev_category = category
        #     prev_answer = answer
        #     await message.answer(answer + f"\nКатегория: {key}")
        #     return
        
        # elif category in [0, 1, 6, 7]:
        #     answer = gen.generate(message.text, category)
        #     answer = post_process(answer)
        #     key = get_key(p.categories_dict, category)
        #     await message.answer(answer + f"\nКатегория: {key}") 
        #     return
        # elif category != prev_category:
        #     if(category != p.categories_dict[sample_category[idx+1]]):
        #         await message.answer("Давайте закончим предыдущий пункт")
        #         await message.answer(prev_answer)
        #         return                
        #     else:
        #         idx += 1
        #         answer = gen.generate(message.text, category)
        #         answer = post_process(answer)
        #         prev_category = category
        #         prev_answer = answer
        #         key = get_key(p.categories_dict, category)
        #         await message.answer(answer + f"\nКатегория: {key}")
        #         return     
        # else:
        #     answer = gen.generate(message.text, category)
        #     answer = post_process(answer)
        #     prev_category = category
        #     prev_answer = answer
        #     key = get_key(p.categories_dict, category)
        #     await message.answer(answer + f"\nКатегория: {key}") 
        #     return


        # if category in [0, 1, 6, 7]:
        #     answer = gen.generate(message.text, category)
        #     answer = post_process(answer)
        #     key = get_key(p.categories_dict, category)
        #     await message.answer(answer)# + f"\nКатегория: {key}")
        #     return
        
        # if(prev_category != -1):
        #     if (prev_category != category)  and (category != p.categories_dict[sample_category[idx+1]]):
        #             await message.answer("Давайте закончим предыдущий пункт")
        #             await message.answer(prev_answer)                
        
        #     else:
        #         answer = gen.generate(message.text, category)
        #         answer = post_process(answer)
        #         prev_answer = answer
        #         prev_category = category
        #         key = get_key(p.categories_dict, category)
        #         idx += 1
        #         await message.answer(answer)# + f"\nКатегория: {key}")
        
        # else:
        #     answer = gen.generate(message.text, category)
        #     answer = post_process(answer)
        #     prev_answer = answer
        #     prev_category = category
        #     key = get_key(p.categories_dict, category)
        #     await message.answer(answer)# + f"\nКатегория: {key}")
        
        
        
            
        # elif (prev_category != -1):
        #     if (prev_category != category):
        #         if(category != p.categories_dict[sample_category[idx+1]]):
        #             await message.answer("Давайте закончим предыдущий пункт")
        #             await message.answer(prev_answer)            
        # else:
        #     answer = gen.generate(message.text, category)
        #     answer = post_process(answer)
        #     prev_answer = answer
        #     prev_category = category
        #     key = get_key(p.categories_dict, category)
        #     await message.answer(answer + f"\nКатегория: {key}")

    executor.start_polling(dp, skip_updates=True)
