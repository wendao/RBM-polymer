#pragma once

#include <iostream>
#include <cmath>
using namespace std;

//sigmoid(13.8155) = 0.999999
//sigmoid(-13.8155) = 1e-6

namespace utils {

  float uniform(float min, float max) {
    return rand() / (RAND_MAX + 1.0) * (max - min) + min;
  }

  int uniformint(int min, int max) {
    return int( rand() / (RAND_MAX + 1.0) * (max - min) + min );
  }

  int binomial(int n, float p) {
    if(p < 0 || p > 1) return 0;

    int c = 0;
    float r;

    for(int i=0; i<n; i++) {
      r = rand() / (RAND_MAX + 1.0);
      if (r < p) c++;
    }

    return c;
  }

  //1.0/sigmoid(-x) = 1.0+exp(x)
  float sigmoid(float x) {
    if (x>13.8155) return 0.999999;
    if (x<-13.8155) return 1.0e-6;
    return 1.0 / (1.0 + exp(-x));
  }

  //a*b dot bxc -> a*c
  void dot(float **MatC, float **MatA, float **MatB, int a, int b, int c) {
    for (int i=0; i<a; i++) {
      for (int j=0; j<c; j++) {
        MatC[i][j] = 0.0;
        for (int k=0; k<b; k++) {
          MatC[i][j] += MatA[i][k]*MatB[k][j];
        }
      }
    }
  }

  void dot(float **MatC, int **MatA, float **MatB, int a, int b, int c) {
    for (int i=0; i<a; i++) {
      for (int j=0; j<c; j++) {
        MatC[i][j] = 0.0;
        for (int k=0; k<b; k++) {
          MatC[i][j] += MatA[i][k]*MatB[k][j];
        }
      }
    }
  }

}
