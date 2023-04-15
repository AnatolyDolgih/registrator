# classificator
# generator
# dialog_control

def get_replic():
    str_ = input()
    return str_;  

def post_process(replic):
    text = replic.lstrip().rstrip().capitalize()
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    return text
  
# импортировали модуль классификатора
import registrator.classification as cls
import registrator.generation as gn

# инициализация классификатора
#classificator = cls.Classificator()
print(gn.pathToModels)
generator = gn.Registrator(gn.pathToModels + "/vocab.pt", 
                           gn.pathToModels + "/generationModel.pth")

replic = ""
while(replic != "exit"):
    # фаза 1 - получение реплики пользователя
    print(f"Введите реплику")
    replic = get_replic()
    answer = generator.generateRegistratorAnswer(replic)
    # фаза 2 - классифицировать полученную реплику
    # (replic) --> [ classificator ] --> (category)
    #category = classificator.classify(replic)
    answer = post_process(answer)
    print(f"{answer}")

