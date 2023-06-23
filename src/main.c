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
  Value b = initValue(6.8813735870195432, "b");
  Value x1w1 = multiply(x1, w1);
  x1w1.label = "x1w1";
  Value x2w2 = multiply(x2, w2);
  x2w2.label = "x2w2";
  Value x1w1_x2w2 = add(x1w1, x2w2);
  x1w1_x2w2.label = "x1w1+x2w2";

  // sets self to x1w1_x2w2, other to b, out to n
  Value n = add(x1w1_x2w2, b);
  n.label = "n";

  // sets self to n, other is not defined, output to o
  Value o = tanhv(n);
  o.label = "o";
  o.grad = 1.0;
  printValue(o);

  // self, other, out
  o.backward(&n, &n, &o);
  printValue(n);

  n.backward(&x1w1_x2w2, &b, &n);
  printValue(x1w1_x2w2);
  printValue(b);

  x1w1_x2w2.backward(&x1w1, &x2w2, &x1w1_x2w2);
  printValue(x1w1);
  printValue(x2w2);

  x1w1.backward(&x1, &w1, &x1w1);
  printValue(x1);
  printValue(w1);

  x2w2.backward(&x2, &w2, &x2w2);
  printValue(x2);
  printValue(w2);
}

int main() {
  // manualBackprop();
  // plot(tanhf);
  testNN();
}
