digit = {"1": "один", "2": "два", "3": "три", "4": "четыре", "5": "пять", 
         "6": "шесть", "7": "семь", "8": "восемь", "9": "девять", "10": "десять"}

data = "Добрый вечер"
data1 = data.split(" ")
new_data = ""
for i in data1:
    if i in digit.keys():
         i = digit[i]
    new_data += str(i)
    new_data += " "
    
print(new_data)