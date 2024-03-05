from datetime import datetime

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

def write_log(message, service):
    print(f'LawSage v0.2-{service}-[{formatted_datetime}]> '+ str(message))
