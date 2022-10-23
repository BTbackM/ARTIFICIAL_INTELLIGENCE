#include <iostream>

int main(int argc, char *argv[]) {
  int N;

  if(argc == 1) {
    N = 2;
  }
  else {
    N = atoi(argv[1]);
  }

  printf("Hello World \n");

  return 0;
}

