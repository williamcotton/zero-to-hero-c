#include "memory.h"
#include "mlp.h"
#include "mse_loss.h"
#include "plot.h"
#include "value.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float quadratic(float x) { return 7 * powf(x, 2) - 4 * x + 5; }

#define NUM_POINTS 40

#define ONE_MEG (1024 * 1024)
#define ONE_K (1024)

void print_banner(const char *title) {
  int len = (int)strlen(title);
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

void print_subheader(const char *subheader_text, char *text) {
  printf("\n\033[34m");
  printf("%s: ", subheader_text);
  printf("\033[0m");
  printf("%s\n\033[34m", text);
  int len = (int)strlen(text) + (int)strlen(subheader_text) + 2;
  for (int i = 0; i < len; i++)
    printf("-");
  printf("\033[0m\n");
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

  nm_t *nm = nm_create((size_t)(ONE_K * 0.75));
  nm_print(nm);

  // inputs x1,x2
  Value *x1 = value_create(2.0, nm);
  nm_print(nm);
  Value *x2 = value_create(0.0, nm);
  nm_print(nm);
  // weights w1,w2
  Value *w1 = value_create(-3.0, nm);
  nm_print(nm);
  Value *w2 = value_create(1.0, nm);
  nm_print(nm);
  // bias of the neuron
  Value *b = value_create(6.8813735870195432, nm);
  nm_print(nm);
  // x1*w1 + x2*w2 + b
  Value *x1w1 = value_multiply(x1, w1, nm);
  nm_print(nm);
  Value *x2w2 = value_multiply(x2, w2, nm);
  nm_print(nm);
  Value *x1w1x2w2 = value_add(x1w1, x2w2, nm);
  nm_print(nm);
  Value *n = value_add(x1w1x2w2, b, nm);
  nm_print(nm);
  Value *o = value_tanhv(n, nm);
  nm_print(nm);

  // Print the values and gradients of each node
  nm_t *bpnm = nm_create(ONE_K * 9);
  value_backpropagate(o, bpnm);
  nm_free(bpnm);

  value_print(o, 0);

  nm_free(nm);
}

void nn1() {
  print_banner("nn1");

  nm_t *nm = nm_create(ONE_K);

  print_subheader("Creating inputs", "*x[2] = {x0, x1}");
  Value *x[2];
  x[0] = value_create(2.0, nm);
  x[1] = value_create(3.0, nm);
  nm_print(nm);

  print_subheader("Creating neuron", "nin=2, layer_id=0, neuron_id=0, nm");
  Neuron *neuron = neuron_create((neuron_params){
      .nin = 2,
      .layer_id = 0,
      .neuron_id = 0,
      .nm = nm,
  });
  nm_print(nm);
  neuron_print(neuron);
  print_subheader("Calling neuron", "neuron, x, nm");
  Value *result = neuron_call(neuron, x, nm);
  nm_print(nm);
  printf("Output: %f\n", result->data);

  nm_free(nm);
}

void layer1() {
  print_banner("layer1");

  nm_t *nm = nm_create(ONE_K * 4);

  print_subheader("Creating inputs", "*x[2] = {x0, x1}");
  Value *x[2];
  x[0] = value_create(2.0, nm);
  x[1] = value_create(3.0, nm);
  nm_print(nm);

  print_subheader("Creating layer", "nin=2, nout=3, layer_id=0, nm");
  Layer *layer = layer_create((layer_params){
      .nin = 2,
      .nout = 3,
      .layer_id = 0,
      .nm = nm,
  });
  nm_print(nm);
  Value **outs = nm_malloc(nm, sizeof(Value *) * (size_t)layer->nout);
  print_subheader("Calling layer", "layer, x, outs, nm");
  Value **result = layer_call(layer, x, outs, nm);
  nm_print(nm);
  for (int i = 0; i < 3; i++) {
    printf("Output: %f\n", result[i]->data);
  }

  nm_free(nm);
}

void mlp1() {
  print_banner("mlp1");

  nm_t *nm = nm_create(ONE_K * 11);

  Vector *x = value_create_vector((double[]){2.0, 3.0, -1.0}, 3, nm);
  nm_print(nm);

  MLP *mlp = mlp_create((mlp_params){
      .nin = 3,
      .nouts = (int[]){4, 4, 1},
      .nlayers = 3,
      .nm = nm,
  });
  nm_print(nm);

  UNUSED Value *ypred = mlp_call(mlp, x->values, nm);
  nm_print(nm);

  mlp_print(mlp);
  nm_free(nm);
}

void trainingLoop() {
  print_banner("trainingLoop");

  nm_t *trainingNm = nm_create(ONE_K * 2);

  int outputCount = 4;
  int trainingCount = 4;

  size_t baseMemory = (size_t)outputCount * (size_t)trainingCount; // 16

  Vector **xs = nm_malloc(trainingNm, sizeof(Vector *) * (size_t)trainingCount);
  double xs_data[][3] = {
      {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  for (int i = 0; i < trainingCount; i++) {
    xs[i] = value_create_vector(xs_data[i], 3, trainingNm);
  }

  nm_print(trainingNm);

  Vector *labels = value_create_vector((double[]){1.0, -1.0, -1.0, 1.0},
                                       outputCount, trainingNm);

  nm_t *mlpNm = nm_create(ONE_K * baseMemory); // 16k

  MLP *mlp = mlp_create((mlp_params){
      .nin = 3,
      .nouts = (int[]){4, 4, 1},
      .nlayers = 3,
      .nm = mlpNm,
  });

  nm_print(mlpNm);

  int epochsCount = 30;
  double learningRate = 0.05;

  for (int epoch = 0; epoch < epochsCount; epoch++) {
    // allocate memory for this epoch
    nm_t *epochNm = nm_create(ONE_K * baseMemory * 3); // 48k

    // zero gradients
    mlp_zero_grad(mlp);

    Value *mseLoss = mse_loss_create(epochNm);
    for (int i = 0; i < trainingCount; i++) {
      // forward pass: compute predictions.
      Value *ypred = mlp_call(mlp, xs[i]->values, epochNm);
      // compute the loss between predicted and actual outputs.
      mseLoss = mse_loss_call(mseLoss, ypred, labels->values[i], epochNm);
    }

    // backward pass: compute gradients
    value_backpropagate(mseLoss, epochNm);

    // update parameters
    mlp_update_parameters(mlp, learningRate);

    printf("%d %.15lf\n", epoch, mseLoss->data);

    // free up memory allocated for this epoch
    nm_free(epochNm);
  }

  nm_t *predictNm = nm_create(ONE_K * baseMemory * 2); // 32k
  for (int i = 0; i < outputCount; i++) {
    Value *ypred = mlp_call(mlp, xs[i]->values, predictNm);
    printf("ypred[%d]: %.15lf\n", i, ypred->data);
  }
  nm_free(predictNm);
  nm_free(trainingNm);
  nm_free(mlpNm);
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
