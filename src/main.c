#include "plot.h"
#include "value.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_POINTS 40

float f(float x) { return 7 * pow(x, 2) - 4 * x + 5; }

void plot(float (*f)(float)) {
  double xs[NUM_POINTS], ys[NUM_POINTS];

  ys_equal_f_of_xs(ys, f, xs, NUM_POINTS,
                   (range){
                       .start = -5,
                       .end = 5,
                       .step = 0.2,
                   });

  plot_data(xs, ys, NUM_POINTS, "Function Plot");
}

void manualBackprop() {
  Value a = initValue(2.0, "a");
  Value b = initValue(-3.0, "b");
  Value c = initValue(10, "c");
  Value e = multiply(a, b);
  e.label = "e";
  Value d = add(e, c);
  d.label = "d";
  Value f = initValue(-2.0, "f");
  Value L = multiply(d, f);
  L.label = "L";
  printValue(L);
}

void testNN() {
  // inputs x1,x2
  Value x1 = initValue(2.0, "x1");
  Value x2 = initValue(0.0, "x2");
  // weights w1,w2
  Value w1 = initValue(-3.0, "w1");
  Value w2 = initValue(1.0, "w2");
  // bias of the neuron
  Value b = initValue(6.7, "b");
  Value x1w1 = multiply(x1, w1);
  x1w1.label = "x1w1";
  Value x2w2 = multiply(x2, w2);
  x2w2.label = "x2w2";
  Value x1w1_x2w2 = add(x1w1, x2w2);
  x1w1_x2w2.label = "x1w1+x2w2";
  Value n = add(x1w1_x2w2, b);
  n.label = "n";
  Value o = tanhv(n);
  o.label = "o";
  printValue(o);
}

int main() {
  // manualBackprop();
  // plot(tanhf);
  testNN();
}
