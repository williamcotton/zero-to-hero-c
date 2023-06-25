#ifndef NEURALNET_H
#define NEURALNET_H

#include "value.h"

typedef struct Neuron {
  Value **w;
  Value *b;
  int nin;
} Neuron;

Neuron *neuron_create(int nin);
Value *neuron_call(Neuron *neuron, Value **x);
Value **neuron_parameters(Neuron *neuron);
void neuron_free(Neuron *neuron);
void neuron_print(Neuron *neuron);

typedef struct Layer {
  Neuron **neurons;
  int nin;
  int nout;
} Layer;

Layer *layer_create(int nin, int nout);
Value **layer_call(Layer *layer, Value **x, Value **outs);
Value **layer_parameters(Layer *layer);
int layer_nparams(Layer *layer);
void layer_free(Layer *layer);

typedef struct MLP {
  Layer **layers;
  int n;
} MLP;

MLP *mlp_create(int nin, int *nouts, int n);
Value **mlp_call(MLP *mlp, Value **x);
void mlp_update_graph(MLP *mlp);
Value **mlp_parameters(MLP *mlp);
int mlp_nparams(MLP *mlp);
void mlp_free(MLP *mlp);
void mlp_print(MLP *mlp);

#endif
