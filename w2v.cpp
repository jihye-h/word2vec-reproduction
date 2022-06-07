/*
489403 vocabs
onebillion datatset
문장 X
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <string>
#include <cstring>
#include <ctime>
#include <cmath>
#include <random>
#include <thread>
using namespace std;

#define MAX_EXP 6 // 시그모이드 1,0에 가깝다고치는 값

class W2V{ 
public:
    // 단어 개수와 차원 수
    int vocab_size = 489403;
    int dim = 300;
    float sample = 1e-5;

    // (단어 id 순서대로) 각 단어의 속성들
    vector<int> vocabid;
    vector< vector<int> > vocabpoint;
    vector<int> vocabfreq;
    vector<string> vocabcode;
    vector<int> vocabcodelen;
    vector<vector<string> > wordtoid;

    // layers
    vector< vector<float> > embed; // 단어 임베딩 계층 
    vector<float> hidden; // hidden 계층 
    vector<float> hiddenE; // hidden for 역전파
    vector< vector<float> > weight; // 임베딩과 동일하게 초기화
    
    float *expTable;

    int hs = 0, ns = 1;
    int negative = 5;
    int layer1_size = 300;
    float starting_alpha = 0.025; // cbow:0.05, skip-gram:0.025
    int get_window();
    float sigmoid(float x);
    int getNSrandom();

    void GetVocab();
    void CreateTree();

    void InitLayers();
    void InitEmbedding();
    void InitHidden();
    void InitHiddenE();
    void InitUnigramTable();

    void GetTokenizedFile(string file);
    vector< vector <int> > tokenized_file;
    int filesize = 0;
    int sensize = 0;
    void TrainOneFile_cbow(int file);
    void TrainOneFile_skipgram(int file);
    //float *syn0, *syn1, *syn1neg, *expTable;

    void Train(int a);
    void SaveVectors();
    void GetWordtoId();
    void Preprocess();

    const int table_size = 1e8;
    int *table;
};

void W2V::InitUnigramTable() {
  
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocabfreq[a], power);
  i = 0;
  d1 = pow(vocabfreq[a], power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocabfreq[i], power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
  
}

float W2V::sigmoid(float x) {
     float result;
     result = (float)1 / (float) (1 + exp(-x));
     return result;
}

int W2V::get_window(){
  // 시드값을 얻기 위한 random_device 생성.
  random_device rd;
  // random_device 를 통해 난수 생성 엔진을 초기화 한다.
  mt19937 gen(rd());
  // 0 부터 n 까지 균등하게 나타나는 난수열을 생성하기 위해 균등 분포 정의.
  std::uniform_int_distribution<int> dis(1, 5);
  return dis(gen);
}

int W2V::getNSrandom(){
  // 시드값을 얻기 위한 random_device 생성.
  random_device rd;
  // random_device 를 통해 난수 생성 엔진을 초기화 한다.
  mt19937 gen(rd());
  // 0 부터 n 까지 균등하게 나타나는 난수열을 생성하기 위해 균등 분포 정의.
  //std::uniform_int_distribution<int> dis(0, table_size);
  std::uniform_int_distribution<int> dis(0, table_size);
  return dis(gen);
}

void W2V::GetVocab(){ //syn0
    string str;
    ifstream file("list_vocab_freq");
    if (!file){
      cout<<"file not found"<<endl;
    }
    string word;
    int wordfreq;
    int count = 0;
    for (int i = 0; i<vocab_size;i++){ // word id, 빈도수 저장
        file >> word >> wordfreq;
        //cout<< word << " " << wordfreq << '\n';
        vocabid.push_back(i);
        vocabfreq.push_back(wordfreq);
    }
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void W2V::CreateTree() {
  // min1i is the first son, and min2i is the second son; 
  // point[] is used to store the index of all father nodes
  long long a, b, i, min1i, min2i, pos1, pos2, point[40];
  int code[40];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocabfreq[a];
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  vector <int> tmp;
  for (int i = 0;i<40;i++){
    tmp.push_back(0);
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      //point[] stores the index(주소값) of its ancestors from itself to the highest, 루트 제외
      point[i] = b;
      
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocabcodelen.push_back(i);
    vocabpoint.push_back(tmp);
    vocabpoint[a][0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocabcode.push_back("0000000000000000000000000000000000000000");
      vocabcode[a][i - b - 1] = (char)(code[b] + 48);
      vocabpoint[a][i - b] = point[b] - vocab_size;
      
    }
  }
  free(count);
  free(binary);
  free(parent_node);

}

// init_embed.py 로 만든 w2v_randvec(txt) 파일로 부터 임베딩 초기 벡터 만든다.
// preprocess2.py 에서 만드는 것으로 수정함.
// 약 1분 소요.
void W2V::InitEmbedding(){
  ifstream file("w2v_randvec"); // 파이썬으로 초기화한 배열 텍스트파일
  if(!file){
    cout<<"file not found"<<endl;
  }
  
  for (int i = 0; i<vocab_size;i++){
    float num;
    vector<float> tmp;
    vector<float> zeros;
    for (int j = 0; j<300; j++){
        file >> num;
        //cout << num << " ";
        tmp.push_back(num);
        zeros.push_back(0);
    }
    embed.push_back(tmp);
    weight.push_back(zeros);
    vector<float>().swap(tmp);
  }
  cout << "embedding layer initialized!" << "\n";
  //cout << embed[489402][299] << "\n";
  file.close();
}

void W2V::InitHidden(){ // neu1
  //hidden.clear();
  for(int i =0; i<300;i++){
    hidden.push_back(0);
  }
}
void W2V::InitHiddenE(){ // neu1
  //hiddenE.clear();
  for(int i =0; i<300;i++){
    hiddenE.push_back(0);
  }
}

void W2V::InitLayers() {
  InitEmbedding(); // input(단어 임베딩) 계층 초기화. 
  InitHidden();
  InitHiddenE();
}

void W2V::Preprocess(){
    // 실행 시간 측정
    clock_t start, finish;
    double duration;
    start = clock();

    GetVocab(); // 파일로 부터 단어 인덱스와 빈도수 저장를 id, freq에 저장 (순서대로)
    CreateTree();
    InitLayers();

    expTable = (float *)malloc((1000 + 1) * sizeof(float));
    for (int i = 0; i < 1000; i++) {
      expTable[i] = exp((i / (float)1000 * 2 - 1) * MAX_EXP); // Precompute the exp() table
      expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
    if (negative > 0){
      InitUnigramTable();
      cout<<"Negative Sampling, InitUnigramTable() "<<endl;
    }

    // 실행 시간 출력
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout << "전처리 시간 : "<<duration << "초" << endl;

    GetWordtoId();
    FILE *tree = fopen("list_word_code", "wb");
    for(int v = 0; v<vocab_size;v++){
      fprintf(tree, "%s ",wordtoid[v][0].c_str());
      for (int j = 0; j < vocabcodelen[v];j++){
        fprintf(tree, "%c",vocabcode[v][j]);
      }
      fprintf(tree, "\n");
    }
 
}

void W2V::GetTokenizedFile(string filepath){
// tokenized file to array
    filesize = 0;
    tokenized_file.clear();
    ifstream file(filepath);
    if(!file){
      cout << "file not found" << endl;
    }
    
    int word;
    int count = 0;
    vector < int > sen;
    while(1){
      file >> word;
      sen.push_back(word);
      if (word == 0){
        count ++;
        if(sen.size() > 12){
            tokenized_file.push_back(sen);  
            filesize += sen.size();
        } 
        sen.clear();
        if (file.eof()) break;
        else continue;
      }
      if (file.eof()) break;
      //tokenized_file.push_back(word);
    }

    sensize = tokenized_file.size(); // 해당 파일에 포함된 단어 수
    cout<<"   number of sentences : " << sensize <<endl;
    cout<<"   number of words : " << filesize <<endl;
    file.close();
}


void W2V::TrainOneFile_cbow(int file){
    clock_t start = clock(), now;
    
    string filepath;
    if(file<10) filepath = "tokendata/t_news.en-0000"+to_string(file)+"-of-00100";
    else if(file>=10 && file<100) filepath = "tokendata/t_news.en-000"+to_string(file)+"-of-00100";
    else cout<<"wrong file "<<endl;

    GetTokenizedFile(filepath);

    float f = 0, g = 0, ran = 0;
    int index = 0, label = 0, target = 0;
    unsigned long long next_random = 100;
    float alpha = starting_alpha;
    int d = 0, c = 0, p = 0, sentence_length = 12, word = 0, cn;
    // window size : ~5
    int s = 0, t = 0, window=2, context_size = 10;
    for(s = 0; s < sensize; s++){ // 첫 문장부터 마지막 문장까지 (s = 문장 번호)
      //cout << s << " ";
      sentence_length = tokenized_file[s].size();
      for(t = 2; t < sentence_length - 2; t++){
        word = tokenized_file[s][t]; //중심 단어 id
        cn = vocabfreq[word];
        /*
        //subsampling
        if (sample > 0) {
            ran = (sqrt(cn /(sample * 800000000)) + 1) * (sample * 800000000) / cn;
            next_random = next_random * (unsigned long long)25214903917 + 11;
            if (ran < (next_random & 0xFFFF) / (float)65536) continue;
        }
        */
        //cout << t << " ";
        // input -> hidden : hidden(300) 각 자리에 context 단어들 더함.
        window = get_window();
        //cout<<t<<" ";
        for ( d = 0; d<300;d++) hidden[d] = 0;
        for ( d = 0; d<300;d++) hiddenE[d] = 0;
        
        context_size = 0;
        for ( c = t - window; c <= t + window; c++){ // (c = context word)
            if (c==t) continue;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            context_size ++;
            for( d = 0; d < 300; d++){
              hidden[d] += embed[tokenized_file[s][c]][d];
            }
        }
        // 다 더했으면 단어 수 만큼 나눈다.
        if(context_size > 0) for( d = 0; d < 300; d++){
            hidden[d] /= context_size;
        } 
        // hierarchical softmax
        if(hs){
          for( p = 0; p<vocabcodelen[word];p++){
            f = 0;
            // hidden -> output
            index = vocabpoint[word][p];
            for (int d = 0; d < 300; d++){
              f += hidden[d] * weight[index][d];
            }
            if (f <= -MAX_EXP) continue; // 0에 가까움
            else if (f >= MAX_EXP) continue; // 1에 가까움
            else f = expTable[(int)((f + MAX_EXP) * (1000 / MAX_EXP / 2))]; // 시그모이드 테이블에서 f 위치 찾음.
            // 기울기
            if(vocabcode[word][p] == '0') g = alpha*(1-f);
            else if(vocabcode[word][p] == '1') g = alpha*(-1)*f ;
            //if (t % 1000 == 0 && p == 0)cout << g << endl;

            // output -> hidden
            for ( d = 0; d < 300 ; d++){
              hiddenE[d] += g * weight[index][d];
            }
            // (train) hidden -> output
            for ( d = 0; d < 300 ; d++){
              weight[index][d] += g * hidden[d];
            }
          }
        }
        if(ns) for(p = 0; p < negative + 1; p++){
          if(p == 0){ 
            target = word; // 중심 단어의 id
            label = 1;
          }
          else { // target = 랜덤
            target = table[getNSrandom()];
            //if (target == 0) target = getNSrandom() % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          f = 0;
          for ( d = 0 ; d < 300 ; d++) f += hidden[d] * weight[target][d];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (1000/MAX_EXP/2))]) * alpha;
          //cout << g * 100 << " ";
          for ( d = 0; d < 300 ; d++) hiddenE[d] += g * weight[target][d];
          for (d = 0; d < 300 ; d++) weight[target][d] += g * hidden[d];
        }
        //hidden->input
        for ( c = t - window; c <= t + window; c++){ // (c = context word)
          if (c==t) continue;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          for( d = 0; d<300; d++){
            embed[tokenized_file[s][c]][d] += hiddenE[d];//오차 더해준다
          }
        }
      }
    }
    clock_t finish = clock();
    cout<<filepath<<" / word learning done, "<<(finish - start)/ CLOCKS_PER_SEC<<"(초)"<<endl;
}

void W2V::TrainOneFile_skipgram(int file){
    clock_t start = clock(), now;
    
    string filepath;
    if(file<10) filepath = "tokendata/t_news.en-0000"+to_string(file)+"-of-00100";
    else if(file>=10 && file<100) filepath = "tokendata/t_news.en-000"+to_string(file)+"-of-00100";
    else cout<<"wrong file "<<endl;

    GetTokenizedFile(filepath);

    float f = 0, g = 0, ran = 0;
    int index = 0, label = 0, target = 0, next_random = 1000;
    float alpha = starting_alpha;
    int d = 0, c = 0, p = 0, sentence_length = 12, word = 0, cn;
    // window size : ~5
    int s = 0, t = 0, window=2, context_size = 10;
    
    for(s = 0; s < sensize; s++){
      sentence_length = tokenized_file[s].size();
      for(t = 2; t<sentence_length-2;t++){ // 파일의 6번째 단어부터 끝에서 6번째 단어까지 (t = target word)
        word = tokenized_file[s][t]; //중심 단어 id
        cn = vocabfreq[word];
        /*
        //subsampling
        if (sample > 0) {
            ran = (sqrt(cn /(sample * 700000000)) + 1) * (sample * 700000000) / cn;
            //next_random = next_random * (unsigned long long)25214903917 + 11;
            //if (ran < (next_random & 0xFFFF) / (float)65536) continue;
            next_random = getNSrandom();
            if (ran < (next_random & 0xFFFF) / (float)65536) continue;
        }
        */
        //cout << t << " ";
        // input -> hidden : hidden(300) 각 자리에 context 단어들 더함.
        window = get_window();
        //cout<<t<<" ";
        //for ( d = 0; d<300;d++) hidden[d] = 0; // 사실 필요 없다. embed에서 바로 갔다 쓰면 됨.
        for ( d = 0; d<300;d++) hiddenE[d] = 0;
        //for (d = 0; d<300;d++) hidden[d] = embed[tokenized_file[t]][d];
        for ( c = t - window; c <= t + window; c++){ // (c = context word)
            if (c==t) continue;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            for ( d = 0; d<300;d++) hiddenE[d] = 0;
            
            //hierarchical softmax
            
            if(hs){
              for( p = 0; p<vocabcodelen[word];p++){
                f = 0;
                // hidden -> output
                index = vocabpoint[word][p];
                for (int d = 0; d < 300; d++){
                  f += embed[tokenized_file[s][c]][d] * weight[index][d];
                }
                if (f <= -MAX_EXP) continue; // 0에 가까움
                else if (f >= MAX_EXP) continue; // 1에 가까움
                else f = expTable[(int)((f + MAX_EXP) * (1000 / MAX_EXP / 2))]; // 시그모이드 테이블에서 f 위치 찾음.
                // 기울기
                if(vocabcode[word][p] == '0') g = alpha*(1-f);
                else if(vocabcode[word][p] == '1') g = alpha*(-1)*f ;
                //if (t % 1000 == 0 && p == 0)cout << g << endl;

                // output -> hidden
                for ( d = 0; d < 300 ; d++){
                  hiddenE[d] += g * weight[index][d];
                }
                // (train) hidden -> output
                for ( d = 0; d < 300 ; d++){
                  weight[index][d] += g * embed[tokenized_file[s][c]][d];
                }
              }
            }
            
            if(ns) for(p = 0; p < negative + 1; p++){
              if(p == 0){ 
                target = tokenized_file[s][t]; // 중심 단어의 id
                label = 1;
              }
              else { // target = 랜덤
                target = table[getNSrandom()];
                //if (target == 0) target = getNSrandom() % (vocab_size - 1) + 1;
                if (target == t) continue;
                label = 0;
              }
              //l1:c , l2:target
              f = 0;
              for ( d = 0 ; d < 300 ; d++) f += embed[tokenized_file[s][c]][d] * weight[target][d];
              if (f > MAX_EXP) g = (label - 1) * alpha;
              else if (f < -MAX_EXP) g = (label - 0) * alpha;
              else g = (label - expTable[(int)((f + MAX_EXP) * (1000/MAX_EXP/2))]) * alpha;
              for ( d = 0; d < 300 ; d++) hiddenE[d] += g * weight[target][d];
              for (d = 0; d < 300 ; d++) weight[target][d] += g * embed[tokenized_file[s][c]][d];
            }
            //hidden->input
            for (d = 0; d < 300 ; d++) embed[tokenized_file[s][c]][d] += hiddenE[d];
        }
      }
    }
    clock_t finish = clock();
    cout<<filepath<<" / word learning done, "<<(finish - start)/ CLOCKS_PER_SEC<<"(초)"<<endl;
}



void W2V::Train(int a){
  clock_t start = clock();
  if (a == 0) { // Hierarchical Softmax
    cout << "Hierarchical Softmax" <<endl;
    hs = 1;
    ns = 0;
  }
  else if (a==1){ // Negative Sampling
    cout << "Negative Sampling" << endl;
    hs = 0;
    ns = 1;
  }
  
  for (int i = 0; i<100; i++){
    cout<<"CBOW"<<endl;
    TrainOneFile_cbow(i);

    //cout<<"Skip-gram"<<endl;
    //TrainOneFile_skipgram(i);
    if (i % 10 == 0){
      SaveVectors();
      cout<<i<<"번 파일 저장 완료"<<endl;
    }
  
  }
  
  clock_t finish = clock();
  cout<<"*** Training done, "<<(finish - start)/ CLOCKS_PER_SEC<<"(초)"<<endl;
}

void W2V::GetWordtoId(){ // 파일에 저장하기 위해 [word, id] 2차원 배열 만든다.
  vector<string> tmp;
  string word, id;
  ifstream file("list_word_id");
  if(!file){
    cout<<"file not found"<<endl;
  }
  for (int i = 0; i<vocab_size;i++){
    file >> word >> id;
    tmp.push_back(word);
    tmp.push_back(id);
    wordtoid.push_back(tmp);
    tmp.clear();
  }
  cout<<"got wordtoid, "<<wordtoid.size()<<endl;
}

void W2V::SaveVectors(){
  FILE *ft = fopen("trainedvec_skhs","wb");
  FILE *fb = fopen("trainedvecb_skhs", "wb");
  fprintf(ft, "%d %d\n", vocab_size, 300);
  fprintf(fb, "%d %d\n", vocab_size, 300);
  for(int i = 0; i<vocab_size;i++){
    fprintf(ft, "%s ", wordtoid[i][0].c_str());
    fprintf(fb, "%s ", wordtoid[i][0].c_str());
    for(int b = 0; b<300;b++){
      fprintf(ft, "%lf ", embed[i][b]);
      fwrite(&embed[i][b], sizeof(float), 1, fb);
    }
    fprintf(ft,"\n");
    fprintf(fb,"\n");
  }
  fclose(ft);
  fclose(fb);
  cout<<"vector file saved"<<endl;
}



int main(){
    W2V w2v;
    w2v.Preprocess(); // 전처리 (hs, index)
    //w2v.TrainOneFile_cbow(1);
    w2v.Train(0); // hs:0 , ns:1 
    w2v.SaveVectors();
}