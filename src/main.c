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

  mlp_print(mlp);

  Value **out = mlp_call(mlp, x);
  for (int i = 0; i < finalCount; i++) {
    printf("Output: %f\n", out[i]->data);
    value_print(out[i]);
  }
}

/*

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

convert to c in the function below
*/

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

  int training_epochs = 20;

  int nouts[3] = {4, 4, 1};
  MLP *mlp = mlp_create(3, nouts, 3);

  for (int i = 0; i < training_epochs; i++) {
    // forward pass
    printf("Epoch %d\n", i);
    Value *loss = value_create(0.0, NULL);
    for (int j = 0; j < sampleCount; j++) {
      Value **out = mlp_call(mlp, xs[j]);
      // printf("Output: %p\n", out);
      Value *difference = value_subtract(out[0], ys[j]);
      Value *squared_difference = value_power(difference, 2.0);
      loss = value_add(loss, squared_difference);
    }
    printf("%d %.15lf\n", i, loss->data);
    Value *mse_loss = value_divide(loss, value_create(sampleCount, NULL));

    // backwd pass
    value_zero_grad_graph(mse_loss);
    // value_print(mse_loss);

    value_backpropagate_graph(mse_loss);
    // value_print(mse_loss);

    // mlp_print(mlp);

    // update
    // value_update_graph(mse_loss, 0.1);
    mlp_update_graph(mlp);
    // value_print(mse_loss);

    printf("%d %.15lf\n", i, mse_loss->data);
    value_free_graph(mse_loss);
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
