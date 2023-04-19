# classificator
# generator
# dialog_control
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './registrator/')))

def get_replic():
    str_ = input()
    return str_;  

def post_process(replic):
    text = str(replic)
    text = text.lstrip().rstrip().capitalize()
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    return text
  
# импортировали модуль классификатора
import registrator.classification as cls
import registrator.generation as gn


if __name__ == "__main__":
    # инициализация классификатора
    classificator = cls.Classificator()
    print(gn.pathToModels)
    # generator = gn.Registrator(gn.pathToModels + "/vocab.pt", 
    #                            gn.pathToModels + "/generationModel.pth")

    gen = gn.Generator()
    replic = ""
    while(replic != "exit"):
        # фаза 1 - получение реплики пользователя
        print(f"Введите реплику")
        replic = get_replic()
        # фаза 2 - классифицировать полученную реплику
        # (replic) --> [ classificator ] --> (category)
        category = classificator.classify(replic)
        answer = gen.generate(replic, category)
        answer = post_process(answer)
        print(f"{answer}")

