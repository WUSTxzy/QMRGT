from pathlib import Path
import json

GQA_ROOT = './data/'

path = Path(GQA_ROOT + 'origin')
split2name1 = {
    'train': 'train',
    'valid': 'val',
    'testdev': 'testdev',
}

for split, name in split2name1.items():
    with open(path / ("%s_balanced_questions.json" % name)) as f:
        data = json.load(f)
        new_data = []
        for key, datum in data.items():
            new_datum = {
                'question_id': key,
                'img_id': datum['imageId'],
                'sent': datum['question'],
                'semantic': datum['semantic'],
                'structural': datum['types']['structural']
            }
            if 'answer' in datum:
                new_datum['label'] = {datum['answer']: 1.}
            new_data.append(new_datum)
        json.dump(new_data, open("../%s.json" % split, 'w'),
                  indent=4, sort_keys=True)

        
split2name2 = {
   'ood-all': 'ood_testdev_all',
   'ood-head': 'ood_testdev_head',
   'ood-tail': 'ood_testdev_tail',
}

for split, name in split2name2.items():
    with open(path / ("%s.json" % name)) as f:
        data = json.load(f)
        new_data = []
        for key, datum in data.items():
            new_datum = {
                'question_id': key,
                'img_id': datum['imageId'],
                'sent': datum['question'],
                'semantic': datum['semantic'],
                'structural': datum['types']['structural']
            }
            if 'answer' in datum:
                new_datum['label'] = {datum['answer']: 1.}
            new_data.append(new_datum)
        json.dump(new_data, open("../%s.json" % split, 'w'),
                  indent=4, sort_keys=True)
