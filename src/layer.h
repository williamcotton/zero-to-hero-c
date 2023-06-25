#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include "value.h"

typedef struct Layer {
  Neuron **neurons;
  int nin;
  int nout;
} Layer;

Layer *layer_create(int nin, int nout, int layer_id);
Value **layer_call(Layer *layer, Value **x, Value **outs);
Value **layer_parameters(Layer *layer);
int layer_nparams(Layer *layer);
void layer_free(Layer *layer);

#endif
