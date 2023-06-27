#include "memory.h"
#include "mlp.h"
#include "mse_loss.h"
#include "plot.h"
#include "value.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float quadratic(float x) { return 7 * pow(x, 2) - 4 * x + 5; }

#define NUM_POINTS 40

#define ONE_MEG (1024 * 1024)
#define ONE_K (1024)

void print_banner(const char *title) {
  int len = strlen(title);
  int total_len = len + 6; // 3 characters padding on each side
  printf("\n\033[34m");
  for (int i = 0; i < total_len; i++)
    printf(i % 2 == 0 ? "-" : "=");
  printf("\n");
  printf("   %s   \n", title);
  for (int i = 0; i < total_len; i++)
    printf(i % 2 == 0 ? "-" : "=");
  printf("\033[0m\n\n");
}

void plot(float (*f)(float), char *filename) {
  print_banner("plot");

  printf("Plotting function: %s\n", filename);

  double xs[NUM_POINTS], ys[NUM_POINTS];

  plot_gen_ys_f_of_xs(ys, f, xs, NUM_POINTS,
                      (Range){
                          .start = -5,
                          .end = 5,
                          .step = 0.2,
                      });

  plot_data(xs, ys, NUM_POINTS, "Function Plot", filename);
}

void nnGraph() {
  print_banner("nnGraph");

  nm_t *nm = nm_create(ONE_MEG);

  // inputs x1,x2
  Value *x1 = value_create(2.0, "x1", nm);
  Value *x2 = value_create(0.0, "x2", nm);
  // weights w1,w2
  Value *w1 = value_create(-3.0, "w1", nm);
  Value *w2 = value_create(1.0, "w2", nm);
  // bias of the neuron
  Value *b = value_create(6.8813735870195432, "b", nm);
  // x1*w1 + x2*w2 + b
  Value *x1w1 = value_multiply(x1, w1, nm);
  x1w1->label = "x1*w1";
  Value *x2w2 = value_multiply(x2, w2, nm);
  x2w2->label = "x2*w2";
  Value *x1w1x2w2 = value_add(x1w1, x2w2, nm);
  x1w1x2w2->label = "x1*w1 + x2*w2";
  Value *n = value_add(x1w1x2w2, b, nm);
  n->label = "n";
  Value *o = value_tanhv(n, nm);
  o->label = "o";

  // Print the values and gradients of each node
  value_backpropagate(o, nm);
  value_print(o, 0);

  nm_free(nm);
}

void nn1() {
  print_banner("nn1");

  nm_t *nm = nm_create(ONE_MEG);

  Value *x[2];
  x[0] = value_create(2.0, "x0", nm);
  x[1] = value_create(3.0, "x1", nm);

  Neuron *neuron = neuron_create((neuron_params){
      .nin = 2,
      .layer_id = 0,
      .neuron_id = 0,
      .nm = nm,
  });
  neuron_print(neuron);
  Value *result = neuron_call(neuron, x, nm);
  value_print(result, 0);
  printf("Output: %f\n", result->data);

  nm_free(nm);
}

void layer1() {
  print_banner("layer1");

  nm_t *nm = nm_create(ONE_MEG);

  Value *x[2];
  x[0] = value_create(2.0, "x0", nm);
  x[1] = value_create(3.0, "x1", nm);

  Layer *layer = layer_create((layer_params){
      .nin = 2,
      .nout = 3,
      .layer_id = 0,
      .nm = nm,
  });
  Value **outs = nm_malloc(nm, sizeof(Value *) * layer->nout);
  Value **result = layer_call(layer, x, outs, nm);
  for (int i = 0; i < 3; i++) {
    printf("Output: %f\n", result[i]->data);
    value_print(result[i], 0);
  }

  nm_free(nm);
}

void mlp1() {
  print_banner("mlp1");

  nm_t *nm = nm_create(ONE_K * 16);

  Vector *x = value_create_vector((double[]){2.0, 3.0, -1.0}, 3, nm);

  MLP *mlp = mlp_create((mlp_params){
      .nin = 3,
      .nouts = (int[]){4, 4, 1},
      .nlayers = 3,
      .nm = nm,
  });

  mlp_print(mlp);
  Value *ypred = mlp_call(mlp, x->values, nm);
  value_print(ypred, 0);
  nm_free(nm);
}

void trainingLoop() {
  print_banner("trainingLoop");

  nm_t *nm = nm_create(ONE_K * 16);

  int outputCount = 4;

  Vector **xs = nm_malloc(nm, sizeof(Vector *) * outputCount);
  double xs_data[][3] = {
      {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  for (int i = 0; i < outputCount; i++) {
    xs[i] = value_create_vector(xs_data[i], 3, nm);
  }

  Vector *ys =
      value_create_vector((double[]){1.0, -1.0, -1.0, 1.0}, outputCount, nm);

  MLP *mlp = mlp_create((mlp_params){
      .nin = 3,
      .nouts = (int[]){4, 4, 1},
      .nlayers = 3,
      .nm = nm,
  });

  int epochsCount = 30;
  double learningRate = 0.05;

  for (int epoch = 0; epoch < epochsCount; epoch++) {
    // forward pass
    nm_t *epochNm = nm_create(ONE_K * 48);

    Value *mseLoss = mse_loss_create(epochNm);
    for (int i = 0; i < outputCount; i++) {
      Value *ypred = mlp_call(mlp, xs[i]->values, epochNm);
      mseLoss = mse_loss_call(mseLoss, ypred, ys->values[i], epochNm);
      if (epoch == epochsCount - 1) {
        printf("ypred[%d]: %.15lf\n", i, ypred->data);
      }
    }

    // backward pass
    mlp_zero_grad(mlp);
    value_backpropagate(mseLoss, epochNm);

    // update parameters
    mlp_update_parameters(mlp, learningRate);

    printf("%d %.15lf\n", epoch, mseLoss->data);

    nm_free(epochNm);
  }

  nm_free(nm);
}

int main() {

  plot(tanhf, "images/tanhf.png");
  plot(quadratic, "images/quadratic.png");
  nnGraph();
  nn1();
  layer1();
  mlp1();
  trainingLoop();
}
