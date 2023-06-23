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

void mlp() {
  // Define input and output data
  float xs[4][3] = {
      {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  float ys[4] = {1.0, -1.0, -1.0, 1.0};

  // Create MLP with 3 input neurons, 2 hidden layers with 4 neurons each, and 1
  // output neuron
  int nouts[3] = {4, 4, 1};
  MLP *mlp = mlp_create(3, nouts, 3);

  // Train MLP for 20 epochs
  for (int k = 0; k < 20; k++) {
    // Forward pass
    float loss = 0.0;
    for (int i = 0; i < 4; i++) {
      Value *x[3];
      for (int j = 0; j < 3; j++) {
        x[j] = initValue(xs[i][j], NULL);
      }
      Value *yout = mlp_call(mlp, x[0]);
      loss += pow(yout->data - ys[i], 2);
      // freeValue(yout);
      for (int j = 0; j < 3; j++) {
        // freeValue(x[j]);
      }
    }

    // Backward pass
    for (int i = 0; i < mlp->n; i++) {
      Layer *layer = mlp->layers[i];
      for (int j = 0; j < layer->nout; j++) {
        Neuron *neuron = layer->neurons[j];
        for (int k = 0; k < neuron->nin; k++) {
          neuron->w[k]->grad = 0.0;
        }
        neuron->b->grad = 0.0;
      }
    }
    backpropagateGraph(initValue(loss, NULL));

    // Update weights and biases
    for (int i = 0; i < mlp->n; i++) {
      Layer *layer = mlp->layers[i];
      for (int j = 0; j < layer->nout; j++) {
        Neuron *neuron = layer->neurons[j];
        for (int k = 0; k < neuron->nin; k++) {
          neuron->w[k]->data += -0.1 * neuron->w[k]->grad;
        }
        neuron->b->data += -0.1 * neuron->b->grad;
      }
    }

    printf("%d %f\n", k, loss);
  }

  // Free MLP
  // mlp_free(mlp);
}

int main() {
  // manualBackprop();
  // plot(tanhf);
  // nnGraph();
  mlp();
}
