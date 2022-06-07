using namespace std;

#include <iostream>
#include <thread>
#include <unistd.h>
#include <time.h>
using std::thread;

void func1() {
  for (int i = 0; i < 10; i++) {
    std::cout << "1 \n";
    sleep(60);
  }
}

void func2() {
  for (int i = 0; i < 10; i++) {
    std::cout << "2 \n";
    sleep(60);
  }
}

void func3(int a) {
  for (int i = 0; i < 10; i++) {
    std::cout << "3 \n";
    sleep(60);
  }
}
int main() {
  time_t start, end;
  double result;
  
  start = time(NULL);
  thread t1(func1);
  thread t2(func2);
  thread t3(func3);

  t1.join();
  t2.join();
  t3.join();

  end = time(NULL); // 시간 측정 끝
  result = (double)(end - start);
  printf("%f", result); //결과 출력
}