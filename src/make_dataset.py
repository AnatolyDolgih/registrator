file = open("D://registrator/registrator/data/new_corpus.txt", "w", encoding='cp1251')
with open("D://registrator/registrator/data/RegistrationDialogs.txt", "r", encoding='cp1251') as f:
    for text in f:
        text = text.lstrip().rstrip()
        if text == "NO TRANSLATIONS FOUND":
            text = "Да, но, к сожалению, номер брони я не знаю, и сейчас узнать не смогу."
        file.write(text + '\n')
file.close()