import pandas as pd
import json
import sys

inputted_path = sys.argv[1]

f = open(inputted_path, "r")
data = json.load(f)

df = pd.DataFrame(columns=['id', 'title', 'context', 'question', 'answers'])
dicts_list = []

for i in range(len(data['data'])):
    row_dict = dict.fromkeys(['id', 'title', 'context', 'question', 'answers'])

    item = data['data'][i]['paragraphs'][0]
    row_dict['context'] = item['context']
    qa = item['qas'][0]
    row_dict['question'] =  qa['question']
    row_dict['id'] =  str(i)
    row_dict['title'] =  str(qa['id'])
    ans = qa['answers'][0]
    new_ans_dict = {
        'answer_start' : [ans['answer_start']], 
        "text" : [ans['text']]
    }
    row_dict['answers'] = new_ans_dict

    df = df.append(row_dict, ignore_index=True)
    dicts_list.append(row_dict)

data_dict = {"data" : dicts_list}

# df.set_index("id", inplace=True)

lst = inputted_path.split(".")
lines_json_path = ".".join(lst[:-1]) + ".squad_format.lines." + "json"
nested_json_path = ".".join(lst[:-1]) + ".squad_format.nested." + "json"
csv_path = ".".join(lst[:-1]) + ".squad_format." + "csv"

df.to_json(lines_json_path, orient='records', lines=True)
df.to_csv(csv_path)

f = open(nested_json_path, "w")
json.dump(data_dict, f)