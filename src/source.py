from control import classification as cls
from control import generation as gen
from control import dialog_control as dc

from interface import interface as ifc

if __name__ == "__main__":
    classificator = cls.Classificator("../models/intent_catcher.pt")
    print(classificator.intent_classification("привет"))
    # generator = gen.Generator()
    # dialog_ctrl = dc.DialogControl()
    
    # replic = "af"
    # #  определим категорию высказывания
    # category = classificator.classify(replic)
    # ctrl = dialog_ctrl.getCtrlState(category)
    # answer = generator.getAnswer(replic, category)
    # # str = classificator.classify("Долгих Анатолий Андреевич ")
    # # print(f"категория: {str}")
    # # classificator.find_binary
