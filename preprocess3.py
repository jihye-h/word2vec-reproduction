# 데이터셋 각 파일들 단어 id 로 변경

import os
import pickle
import time

start = time.time()

with open("wordid.pickle", "rb") as fr:
    data = pickle.load(fr)

word_to_id = data['word_to_id']
count = 0
for filename in os.listdir('onebilliondataset'):
    print(filename)
    with open(os.path.join("onebilliondataset", filename),'r') as f:
        count += 1
        text = f.read()
        text = text.replace("\n", " </s> ")
        text = text.replace('\n', " </s> ")
        text = text.replace('  ', " ")
        text = text.replace("  ", " ")
        text = text.lower()
        text = text.split()
        t = open('tokendata/t_'+filename, 'w')
        for word in text:
            if word in word_to_id.keys():
                t.write(f'{word_to_id[word]} ')
        t.close()
print(count)