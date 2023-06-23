#include "neuralnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Neuron *neuron_create(int nin) {
  Neuron *neuron = malloc(sizeof(Neuron));
  neuron->w = malloc(sizeof(Value *) * nin);
  for (int i = 0; i < nin; i++) {
    neuron->w[i] = initValue(
        (float)arc4random_uniform(UINT32_MAX) / UINT32_MAX * 2.0 - 1.0, NULL);
  }
  neuron->b = initValue(
      (float)arc4random_uniform(UINT32_MAX) / UINT32_MAX * 2.0 - 1.0, NULL);
  neuron->nin = nin;
  return neuron;
}

Value *neuron_call(Neuron *neuron, Value **x) {
  Value *act = neuron->b;
  for (int i = 0; i < neuron->nin; i++) {
    act = add(act, multiply(neuron->w[i], x[i]));
  }
  Value *out = tanhv(act);
  return out;
}

void neuron_free(Neuron *neuron) {
  for (int i = 0; i < neuron->nin; i++) {
    freeValue(neuron->w[i]);
  }
  free(neuron->w);
  freeValue(neuron->b);
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
    neuron_print(layer->neurons[i]);
  }
  layer->nin = nin;
  layer->nout = nout;
  return layer;
}

Value **layer_call(Layer *layer, Value **x) {
  Value **outs = malloc(sizeof(Value *) * layer->nout);
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

Value *mlp_call(MLP *mlp, Value *x) {
  Value *out = x;
  for (int i = 0; i < mlp->n; i++) {
    Value **outs = layer_call(mlp->layers[i], &out);
    if (mlp->layers[i]->nout == 1) {
      out = outs[0];
      // free(outs);
    } else {
      out = initValue(0.0, NULL);
      for (int j = 0; j < mlp->layers[i]->nout; j++) {
        out = add(out, outs[j]);
        // freeValue(outs[j]);
      }
      // free(outs);
    }
  }
  return out;
}

void mlp_free(MLP *mlp) {
  for (int i = 0; i < mlp->n; i++) {
    layer_free(mlp->layers[i]);
  }
  free(mlp->layers);
  free(mlp);
}
