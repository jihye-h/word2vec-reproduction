# list_vocab_freq 만들기

import os
import time
from collections import Counter

start = time.time()

# 단어 한 리스트에 모으기
fulltext = []
for filename in os.listdir('dataset/onebilliondataset'):
    print(filename);
    with open(os.path.join("dataset/onebilliondataset", filename), 'r') as f:
        text = f.read()
        text = text.lower() # 소문자로 변경
        text = text.replace('\n', " </s> ")
        fulltext += text.split()
print("time: ",time.time()-start,"  total words : ",len(fulltext))

# 각 단어의 개수 모으고, 5개 미만 단어 삭제
to_del = [] # 삭제할 단어 키값 저장
counter = Counter(fulltext)
for word, freq in counter.items():
    if freq<5:
        to_del.append(word)
for word in to_del:
    del counter[word]

print("time: ",time.time()-start,"  total vocabs : ",len(counter))

# 정렬해서 텍스트로 저장
counter = counter.most_common()
counter = dict((x,y) for x,y in counter)

f = open('list_vocab_freq', 'w')
sfreq = counter['</s>']
f.write(f"</s> {sfreq}\n")
for word, freq in counter.items():
    if word==None : continue
    if word==" " : continue
    if word=="</s>" : continue
    f.write(f"{word} {freq}\n")
f.close()