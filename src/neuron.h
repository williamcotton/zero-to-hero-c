#ifndef NEURON_H
#define NEURON_H

#include "memory.h"
#include "value.h"

typedef struct Neuron {
  Value **w;
  Value *b;
  int nin;
  ValueList *out;
  nm_t *nm;
} Neuron;

typedef struct neuron_params {
  int nin;
  int layer_id;
  int neuron_id;
  nm_t *nm;
} neuron_params;

Neuron *neuron_create(neuron_params params);
Value *neuron_call(Neuron *neuron, Value **x);
Value **neuron_parameters(Neuron *neuron);
void neuron_print(Neuron *neuron);
void neuron_free(Neuron *neuron);
void neuron_free_value_list(Neuron *neuron);

#endif
