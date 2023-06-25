#ifndef NEURON_H
#define NEURON_H

#include "value.h"

typedef struct Neuron {
  Value **w;
  Value *b;
  int nin;
  ValueList *out;
} Neuron;

Neuron *neuron_create(int nin, int layer_id, int neuron_id);
Value *neuron_call(Neuron *neuron, Value **x);
Value **neuron_parameters(Neuron *neuron);
void neuron_print(Neuron *neuron);
void neuron_free(Neuron *neuron);

#endif
