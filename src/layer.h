#ifndef LAYER_H
#define LAYER_H

#include "memory.h"
#include "neuron.h"
#include "value.h"

typedef struct Layer {
  Neuron **neurons;
  int nin;
  int nout;
  nm_t *nm;
} Layer;

typedef struct layer_params {
  int nin;
  int nout;
  int layer_id;
  nm_t *nm;
} layer_params;

Layer *layer_create(layer_params params);
Value **layer_call(Layer *layer, Value **x, Value **outs);
Value **layer_parameters(Layer *layer);
int layer_nparams(Layer *layer);
void layer_free(Layer *layer);

#endif
