# word_to_id, id_to_word => pickle 저장
# list_id_freq, list_word_id => txt 저장
# w2v_randvec => txt 저장

import time
import pickle

start = time.time()

f = open('list_vocab_freq', 'r')
f_id_freq = open('list_id_freq', 'w')
f_word_id = open('list_word_id', 'w')

word_to_id = {}
id_to_word = {}

line = "temp"
i = 0
while(line):
    line = f.readline()
    line = line.split()
    if len(line) == 2:
        f_id_freq.write(f"{i} {line[1]}\n")
        word_to_id[line[0]] = i 
        id_to_word[i] = line[0]
        f_word_id.write(f"{line[0]} {i}\n")
        i += 1

f.close()
f_id_freq.close()
f_word_id.close()

data = {}
data['word_to_id'] = word_to_id
data['id_to_word'] = id_to_word 

with open("wordid.pickle", 'wb') as fw:
    pickle.dump(data, fw)

print(len(word_to_id))
print(len(id_to_word))

print("time : ", time.time()-start)

# w2v_randvec 
import numpy as np
rand = open('w2v_randvec', 'w')
count = 0
for word in word_to_id.keys():
    count += 1
    if (word_to_id[word] % 100 == 0) : 
        print(word_to_id[word])
    vec = np.round(0.01 * np.random.randn(300),6) #가중치 초깃값 0.01
    for i in range(0,300):
        rand.write(f'{vec[i]} ')
    rand.write(f'\n')
rand.close()
print("randvec done, time : ", time.time()-start)
print(count)