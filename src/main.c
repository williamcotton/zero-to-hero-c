#include "mlp.h"
#include "plot.h"
#include "value.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float quadratic(float x) { return 7 * pow(x, 2) - 4 * x + 5; }

#define NUM_POINTS 40

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

  value_print(o, 0);

  value_free(o);
}

void nn1() {
  print_banner("nn1");

  Value *x[2];
  x[0] = value_create(2.0, "x0");
  x[1] = value_create(3.0, "x1");

  Neuron *neuron = neuron_create(2, 0, 0);
  neuron_print(neuron);
  Value *result = neuron_call(neuron, x);
  value_print(result, 0);
  printf("Output: %f\n", result->data);
}

void layer1() {
  print_banner("layer1");

  Value *x[2];
  x[0] = value_create(2.0, "x0");
  x[1] = value_create(3.0, "x1");

  Layer *layer = layer_create(2, 3, 0);
  Value **outs = malloc(sizeof(Value *) * layer->nout);
  Value **result = layer_call(layer, x, outs);
  for (int i = 0; i < 3; i++) {
    printf("Output: %f\n", result[i]->data);
    value_print(result[i], 0);
  }
  free(result);
  layer_free(layer);
}

void mlp1() {
  print_banner("mlp1");

  Value *x[3];
  x[0] = value_create(2.0, "2.0");
  x[1] = value_create(3.0, "3.0");
  x[2] = value_create(-1.0, "-1.0");

  MLP *mlp = mlp_create((mlp_params){
      .nin = 3,
      .nouts = (int[]){4, 4, 1},
      .nlayers = 3,
  });

  mlp_print(mlp);

  ValueList *ypreds = mlp_call(mlp, x);
  ValueList *current = ypreds;
  while (current != NULL) {
    printf("Output: %.15lf\n", current->value->data);
    current = current->next;
  }
  value_list_free(ypreds);
  mlp_free(mlp);
}

void trainingLoop() {
  print_banner("trainingLoop");

  int outputCount = 4;
  double xs_data[][3] = {
      {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  Value ***xs = malloc(sizeof(Value **) * outputCount);
  for (int i = 0; i < outputCount; i++) {
    xs[i] = value_create_vector(xs_data[i], 3);
  }

  Value **ys =
      value_create_vector((double[]){1.0, -1.0, -1.0, 1.0}, outputCount);

  MLP *mlp = mlp_create((mlp_params){
      .nin = 3,
      .nouts = (int[]){4, 4, 1},
      .nlayers = 3,
  });

  int epochs = 30;
  double learningRate = 0.05;

  for (int epoch = 0; epoch < epochs; epoch++) {
    // forward pass
    Value *mse_loss = value_create(0.0, "mse_loss");
    ValueList *outputs_list = NULL;

    for (int i = 0; i < outputCount; i++) {
      ValueList *ypreds = mlp_call(mlp, xs[i]);
      outputs_list = ypreds;

      Value *ypred = ypreds->value;
      Value *diff = value_subtract(ypred, ys[i]);
      Value *loss = value_power(diff, 2);
      mse_loss = value_add(mse_loss, loss);

      outputs_list = value_list_append(outputs_list, diff);
      outputs_list = value_list_append(outputs_list, loss);
      outputs_list = value_list_append(outputs_list, mse_loss);

      if (epoch == epochs - 1) {
        printf("ypred[%d]: %.15lf\n", i, ypred->data);
      }
    }

    // backward pass
    Value **params = mlp_parameters(mlp);
    int paramsCount = mlp_nparams(mlp);
    for (int i = 0; i < paramsCount; i++) {
      params[i]->grad = 0.0;
    }
    value_backpropagate_graph(mse_loss);

    // update parameters
    for (int i = 0; i < paramsCount; i++) {
      params[i]->data += -learningRate * params[i]->grad;
    }

    printf("%d %.15lf\n", epoch, mse_loss->data);

    value_list_free(outputs_list);
  }
  mlp_free(mlp);
  free(xs);
  free(ys);
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
