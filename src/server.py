import socket
from datetime import datetime
import time
import csv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './registrator/')))
import registrator.classification as cls
import registrator.generation as gn
classificator = cls.Classificator()
gen = gn.Generator()

def post_process(replic):
    text = str(replic)
    text = text.lstrip().rstrip().capitalize()
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    return text

def make_log(log_date, log_time, log_exp_time, role, replic):
    f = open("log_txt.txt", "a")
    log = log_date + " | " + log_time + " | " + log_exp_time + " | "\
        + role + " | " + replic + "\n"
    f.write(log)
    f.close()

def make_csv_log(log_date, log_time, log_exp_time, role, replic):
    f = open("log_csv.csv", "a", newline = '') 
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
     
import sys

digit = {"1": "один", "2": "два", "3": "три", "4": "четыре", "5": "пять", 
         "6": "шесть", "7": "семь", "8": "восемь", "9": "девять", "10": "десять"}

while True:
    start_time = datetime.now()

    f = open("log_csv.csv", "a", newline = '') 
    fieldnames = ['log_date', 'log_time', 'log_exp_time', 'role', 'replic']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.close()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    host = socket.gethostname()
    print(host)
    server_sock.bind(("10.4.34.123", 1080))
    server_sock.listen(5)
    print("Server was run")
    client_sock, addr = server_sock.accept()
    print(f"connection from: {addr}")
    count = 0
    
    while True:

        data = client_sock.recv(10000).decode()
        
        log_date, log_time, log_exp_time = get_logs(start_time)
        
        if not data:
            break
        count += 1
        
            
        make_log(log_date, log_time, log_exp_time, "R", str(data))
        make_csv_log(log_date, log_time, log_exp_time, "R", str(data))
        print(f"{count} >> recieved message: {str(data)}")        

        
        category = classificator.classify(str(data))
        answer = gen.generate(str(data), category)
        answer = post_process(answer)
        
        log_date, log_time, log_exp_time = get_logs(start_time)
        make_log(log_date, log_time, log_exp_time, "V", str(answer))
        make_csv_log(log_date, log_time, log_exp_time, "V", str(answer))

        client_sock.sendall(answer.encode('utf-8'))
        print(f"{count} >> send message: {answer}")

    print(count)
    client_sock.close()
    server_sock.close()

