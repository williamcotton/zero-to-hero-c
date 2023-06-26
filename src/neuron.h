#ifndef NEURON_H
#define NEURON_H

#include "value.h"

typedef struct Neuron {
  Value **w;
  Value *b;
  int nin;
  ValueList *out;
} Neuron;

Neuron *neuron_create(int nin, UNUSED int layer_id, UNUSED int neuron_id);
Value *neuron_call(Neuron *neuron, Value **x);
Value **neuron_parameters(Neuron *neuron);
void neuron_print(Neuron *neuron);
void neuron_free(Neuron *neuron);
void neuron_free_value_list(Neuron *neuron);

#endif
