import xmltodict
import pandas as pd
with open('corpus/smokers_surrogate_train_all_version2.xml') as fd:
    doc = xmltodict.parse(fd.read())
list_recs = []
pure_texts = ''
for rec in doc['ROOT']['RECORD']:
    record = {}
    record['STATUS'] = rec['SMOKING']['@STATUS']
    text = rec['TEXT']
    lines = text.split("\n")
    text = '\n'.join(lines[6:])
    pure_texts = pure_texts + text+'\n'
    record['TEXT'] = text
    list_recs.append(record)
f = open('corpus/i2b2_smoker.txt','w')
f.write(pure_texts)
df = pd.DataFrame(list_recs)
df.to_csv('corpus/i2b2_smoker_status.csv',index=False)

