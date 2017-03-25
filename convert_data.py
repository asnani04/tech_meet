import os
import sys
import json
reload(sys)
sys.setdefaultencoding('utf-8')
j = []
query = 'IBM'
for path, subdirs, files in os.walk('data'):
    for name in files:
        if name[-1]=='n':j.append(os.path.join(path, name))
try:
    os.mkdir(query)
except:
    print ("file exists")
for jf in j:
    with open(jf)as json_file:
        data = json.load(json_file)
    with open(query+'/'+data['title']+'.txt','w')as x:
        x.write(data['text'])
