#include "layer.h"
#include "neuron.h"
#include <stdlib.h>

Layer *layer_create(layer_params params) {
  Layer *layer = nm_malloc(params.nm, sizeof(Layer));
  layer->nm = params.nm;
  layer->neurons = nm_malloc(params.nm, sizeof(Neuron *) * params.nout);
  for (int i = 0; i < params.nout; i++) {
    layer->neurons[i] = neuron_create((neuron_params){
        .nin = params.nin,
        .layer_id = params.layer_id,
        .neuron_id = i,
        .nm = params.nm,
    });
  }
  layer->nin = params.nin;
  layer->nout = params.nout;
  return layer;
}

Value **layer_call(Layer *layer, Value **x, Value **outs) {
  for (int i = 0; i < layer->nout; i++) {
    outs[i] = neuron_call(layer->neurons[i], x);
  }
  return outs;
}

int layer_nparams(Layer *layer) {
  return layer->nin * layer->nout + layer->nout;
}

Value **layer_parameters(Layer *layer) {
  Value **params = nm_calloc(layer->nm, layer_nparams(layer), sizeof(Value *));
  int idx = 0;
  for (int i = 0; i < layer->nout; i++) {
    Value **neuron_params = neuron_parameters(layer->neurons[i]);
    for (int j = 0; j < layer->nin + 1; j++) {
      params[idx++] = neuron_params[j];
    }
    free(neuron_params);
  }
  return params;
}

void layer_free(Layer *layer) {
  for (int i = 0; i < layer->nout; i++) {
    neuron_free(layer->neurons[i]);
  }
}
