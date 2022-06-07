#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

int filesize = 0;
vector <vector <int > > tokenized_file;


void GetTokenizedFile(string filepath){
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

    int sen_size = tokenized_file.size(); // 해당 파일에 포함된 단어 수
    cout<<"   number of sentences : " << sen_size <<endl;
    cout<<"   number of words : " << filesize <<endl;
    file.close();
}

int main(){
    GetTokenizedFile("tokendata/t_news.en-00000-of-00100");
}