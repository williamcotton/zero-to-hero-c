#include "neuralnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Neuron *neuron_create(int nin) {
  Neuron *neuron = malloc(sizeof(Neuron));
  neuron->w = malloc(sizeof(Value *) * nin);
  for (int i = 0; i < nin; i++) {
    neuron->w[i] = value_create(
        (float)arc4random_uniform(UINT32_MAX) / UINT32_MAX * 2.0 - 1.0, NULL);
  }
  neuron->b = value_create(
      (float)arc4random_uniform(UINT32_MAX) / UINT32_MAX * 2.0 - 1.0, NULL);
  neuron->nin = nin;
  return neuron;
}

Value *neuron_call(Neuron *neuron, Value **x) {
  Value *act = neuron->b;
  for (int i = 0; i < neuron->nin; i++) {
    act = value_add(act, value_multiply(neuron->w[i], x[i]));
  }
  Value *out = value_tanhv(act);
  return out;
}

void neuron_free(Neuron *neuron) {
  for (int i = 0; i < neuron->nin; i++) {
    value_free(neuron->w[i]);
  }
  free(neuron->w);
  value_free(neuron->b);
  free(neuron);
}

void neuron_print(Neuron *neuron) {
  printf("Neuron:\n");
  printf("  w: ");
  for (int i = 0; i < neuron->nin; i++) {
    printf("%f ", neuron->w[i]->data);
  }
  printf("\n");
  printf("  b: %f\n", neuron->b->data);
}

Layer *layer_create(int nin, int nout) {
  Layer *layer = malloc(sizeof(Layer));
  layer->neurons = malloc(sizeof(Neuron *) * nout);
  for (int i = 0; i < nout; i++) {
    layer->neurons[i] = neuron_create(nin);
  }
  layer->nin = nin;
  layer->nout = nout;
  return layer;
}

Value **layer_call(Layer *layer, Value **x, Value **outs) {
  for (int i = 0; i < layer->nout; i++) {
    outs[i] = neuron_call(layer->neurons[i], x);
  }
  return outs;
}

void layer_free(Layer *layer) {
  for (int i = 0; i < layer->nout; i++) {
    neuron_free(layer->neurons[i]);
  }
  free(layer->neurons);
  free(layer);
}

MLP *mlp_create(int nin, int *nouts, int n) {
  MLP *mlp = malloc(sizeof(MLP));
  mlp->layers = malloc(sizeof(Layer *) * n);
  for (int i = 0; i < n; i++) {
    mlp->layers[i] = layer_create(i == 0 ? nin : nouts[i - 1], nouts[i]);
  }
  mlp->n = n;
  return mlp;
}

void mlp_print(MLP *mlp) {
  printf("\n=-=-=-=-=\n   MLP\n=-=-=-=-=\n");
  for (int i = 0; i < mlp->n; i++) {
    printf("\nLayer %d: nin=%d nout=%d\n=============================\n", i,
           mlp->layers[i]->nin, mlp->layers[i]->nout);
    for (int j = 0; j < mlp->layers[i]->nout; j++) {
      neuron_print(mlp->layers[i]->neurons[j]);
    }
    printf("-----------------------------\n\n");
  }
}

Value **mlp_call(MLP *mlp, Value **x) {
  Value **xs = x;
  for (int i = 0; i < mlp->n; i++) {
    Value **outs = malloc(sizeof(Value *) * mlp->layers[i]->nout);
    xs = layer_call(mlp->layers[i], xs, outs);
  }
  return xs;
}

void mlp_update_graph(MLP *mlp) {
  double learning_rate = 0.01;
  for (int i = 0; i < mlp->n; i++) {
    for (int j = 0; j < mlp->layers[i]->nout; j++) {
      for (int k = 0; k < mlp->layers[i]->neurons[j]->nin; k++) {
        mlp->layers[i]->neurons[j]->w[k]->data -=
            mlp->layers[i]->neurons[j]->w[k]->grad;
      }
      mlp->layers[i]->neurons[j]->b->data -=
          learning_rate * mlp->layers[i]->neurons[j]->b->grad;
    }
  }
}

void mlp_free(MLP *mlp) {
  for (int i = 0; i < mlp->n; i++) {
    layer_free(mlp->layers[i]);
  }
  free(mlp->layers);
  free(mlp);
}
