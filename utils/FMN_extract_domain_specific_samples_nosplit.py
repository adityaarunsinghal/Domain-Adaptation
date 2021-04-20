import json
import argparse
import os

parser = argparse.ArgumentParser(description='Extract extractive domains specific questions answering samples from MS MARCO')
parser.add_argument('--marco', help='path to MS MARCO training set')
parser.add_argument('--out_dir', help='path to save domain specific datasets')
parser.add_argument('--lookup_table', help='path to the lookup table')
args = parser.parse_args()

# Load lookup table
# {query_id: {domain:str, split:str}}
with open(args.lookup_table, 'r') as fp:
    domain_table = json.load(fp)

# Load MS MARCO training set
with open(args.marco, 'r') as fp:
    marco = json.load(fp)
    rev_query_id = dict()
    for query_id in marco['query']:
        rev_query_id[marco['query_id'][query_id]] = query_id

domains = {'finance': {'train': [], 'dev': [], 'test': []},
           'law': {'train': [], 'dev': [], 'test': []},
           'music': {'train': [], 'dev': [], 'test': []},
           'film': {'train': [], 'dev': [], 'test': []},
           'biomedical': {'train': [], 'dev': [], 'test': []},
           'computing': {'train': [], 'dev': [], 'test': []}}

# Convert extractive domain specific samples into the SQUAD format

domain = "film"

for split in domain_table[domain]:
    for id in domain_table[domain][split]:
        qid, pid = map(int, id.split('_'))
        query_id = rev_query_id[qid]
        context = marco['passages'][query_id][pid]['passage_text'].strip()
        datum = None
        for answer in marco['answers'][query_id]:
            answer = answer.strip()
            if len(answer) > 0:
                # Find answer position
                answer_start = context.lower().find(answer.lower())
                if answer_start >= 0:
                    answer = context[answer_start:answer_start + len(answer)]
                    if datum is None:
                        # Create a QA sample in SQUAD format
                        datum = {
                            'paragraphs': [{
                                'context': context,
                                'qas': [{
                                    'answers': [{'answer_start': answer_start, 'text': answer}],
                                    'question': marco['query'][query_id],
                                    'id': id
                                }]
                            }]
                        }
                    else:  # Handle multiple-answers case
                        datum['paragraphs'][0]['qas'][0]['answers'].append(
                            {'answer_start': answer_start,
                                'text': answer}
                        )
        domains[domain]["train"].append(datum)

# Save
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
with open(os.path.join(args.out_dir, 'squad.film.all.json'), 'w') as fp:
    json.dump({'data': domains[domain]["train"]}, fp)
