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

void nnGraph() {
  // inputs x1,x2
  Value *x1 = initValue(2.0, "x1");
  Value *x2 = initValue(0.0, "x2");
  // weights w1,w2
  Value *w1 = initValue(-3.0, "w1");
  Value *w2 = initValue(1.0, "w2");
  // bias of the neuron
  Value *b = initValue(6.8813735870195432, "b");
  // x1*w1 + x2*w2 + b
  Value *x1w1 = multiply(x1, w1);
  x1w1->label = "x1*w1";
  Value *x2w2 = multiply(x2, w2);
  x2w2->label = "x2*w2";
  Value *x1w1x2w2 = add(x1w1, x2w2);
  x1w1x2w2->label = "x1*w1 + x2*w2";
  Value *n = add(x1w1x2w2, b);
  n->label = "n";
  Value *o = tanhv(n);
  o->label = "o";

  // Print the values and gradients of each node

  backpropagateGraph(o);

  printValue(o);

  freeValue(o);
}

int main() {
  // manualBackprop();
  // plot(tanhf);
  nnGraph();
}
