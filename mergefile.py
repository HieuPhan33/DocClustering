#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import re
import io
import codecs

outfile = open('corpus/merged.txt','w',encoding='utf8')
with zipfile.ZipFile('corpus/corpus.zip', "r") as z:
    for name in z.namelist():
        with z.open(name) as f:
            for _,line in enumerate(io.TextIOWrapper(f, encoding='utf-8')):
                line = re.sub("-","_",str(line))
                outfile.write(line)



