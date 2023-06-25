#include "neuralnet.h"
#include "plot.h"
#include "value.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_POINTS 40

float f(float x) { return 7 * pow(x, 2) - 4 * x + 5; }

void plot(float (*f)(float)) {
  double xs[NUM_POINTS], ys[NUM_POINTS];

  plot_gen_ys_f_of_xs(ys, f, xs, NUM_POINTS,
                      (Range){
                          .start = -5,
                          .end = 5,
                          .step = 0.2,
                      });

  plot_data(xs, ys, NUM_POINTS, "Function Plot", "images/plot.png");
}

void nnGraph() {
  // inputs x1,x2
  Value *x1 = value_create(2.0, "x1");
  Value *x2 = value_create(0.0, "x2");
  // weights w1,w2
  Value *w1 = value_create(-3.0, "w1");
  Value *w2 = value_create(1.0, "w2");
  // bias of the neuron
  Value *b = value_create(6.8813735870195432, "b");
  // x1*w1 + x2*w2 + b
  Value *x1w1 = value_multiply(x1, w1);
  x1w1->label = "x1*w1";
  Value *x2w2 = value_multiply(x2, w2);
  x2w2->label = "x2*w2";
  Value *x1w1x2w2 = value_add(x1w1, x2w2);
  x1w1x2w2->label = "x1*w1 + x2*w2";
  Value *n = value_add(x1w1x2w2, b);
  n->label = "n";
  Value *o = value_tanhv(n);
  o->label = "o";

  // Print the values and gradients of each node

  value_backpropagate_graph(o);

  value_print(o);

  value_free(o);
}

void nn1() {
  Value *x[2];
  x[0] = value_create(2.0, "x0");
  x[1] = value_create(3.0, "x1");

  Neuron *neuron = neuron_create(2);
  neuron_print(neuron);
  Value *result = neuron_call(neuron, x);
  value_print(result);
  printf("Output: %f\n", result->data);
}

void layer1() {
  Value *x[2];
  x[0] = value_create(2.0, "x0");
  x[1] = value_create(3.0, "x1");

  Layer *layer = layer_create(2, 3);
  Value **outs = malloc(sizeof(Value *) * layer->nout);
  Value **result = layer_call(layer, x, outs);
  for (int i = 0; i < 3; i++) {
    printf("Output: %f\n", result[i]->data);
    value_print(result[i]);
  }
}

void mlp1() {
  Value *x[3];
  x[0] = value_create(2.0, NULL);
  x[1] = value_create(3.0, NULL);
  x[2] = value_create(-1.0, NULL);

  int finalCount = 1;

  int nouts[3] = {4, 4, finalCount};
  MLP *mlp = mlp_create(3, nouts, 3);

  // mlp_print(mlp);

  Value **out = mlp_call(mlp, x);
  for (int i = 0; i < finalCount; i++) {
    printf("Output: %.15lf\n", out[i]->data);
    // value_print(out[i]);
  }
}

void trainingLoop() {
  int sampleCount = 4;
  Value ***xs = malloc(sizeof(Value **) * sampleCount);
  xs[0] = malloc(sizeof(Value *) * 3);
  xs[0][0] = value_create(2.0, NULL);
  xs[0][1] = value_create(3.0, NULL);
  xs[0][2] = value_create(-1.0, NULL);
  xs[1] = malloc(sizeof(Value *) * 3);
  xs[1][0] = value_create(3.0, NULL);
  xs[1][1] = value_create(-1.0, NULL);
  xs[1][2] = value_create(0.5, NULL);
  xs[2] = malloc(sizeof(Value *) * 3);
  xs[2][0] = value_create(0.5, NULL);
  xs[2][1] = value_create(1.0, NULL);
  xs[2][2] = value_create(1.0, NULL);
  xs[3] = malloc(sizeof(Value *) * 3);
  xs[3][0] = value_create(1.0, NULL);
  xs[3][1] = value_create(1.0, NULL);
  xs[3][2] = value_create(-1.0, NULL);

  Value **ys = malloc(sizeof(Value *) * 4);
  ys[0] = value_create(1.0, NULL);
  ys[1] = value_create(-1.0, NULL);
  ys[2] = value_create(-1.0, NULL);
  ys[3] = value_create(1.0, NULL);

  int nouts[3] = {4, 4, 1};
  MLP *mlp = mlp_create(3, nouts, 3);

  int epochs = 20;
  double learningRate = 0.01;

  for (int i = 0; i < epochs; i++) {
    // forward pass
    Value *lossSum = value_create(0.0, NULL);
    for (int i = 0; i < sampleCount; i++) {
      Value *ypred = mlp_call(mlp, xs[i])[0];
      Value *loss = value_power(value_subtract(ypred, ys[i]), 2);
      lossSum = value_add(lossSum, loss);
      printf("ypred[%d]: %.15lf\n", i, ypred->data);
    }

    // backward pass
    value_zero_grad_graph(lossSum);
    value_backpropagate_graph(lossSum);

    // update parameters
    Value **allValues = mlp_parameters(mlp);
    for (int i = 0; i < mlp_nparams(mlp); i++) {
      allValues[i]->data += -learningRate * allValues[i]->grad;
    }

    printf("%d %.15lf\n", i, lossSum->data);
  }
  mlp_free(mlp);
  free(xs);
  free(ys);
}

int main() {
  // plot(tanhf);
  // nnGraph();
  // nn1();
  // layer1();
  // mlp1();
  trainingLoop();
}
