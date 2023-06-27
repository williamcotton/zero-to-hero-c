#include "neuron.h"
#include "memory.h"
#include <stdio.h>
#include <stdlib.h>

Neuron *neuron_create(neuron_params params) {
  Neuron *neuron = nm_malloc(params.nm, sizeof(Neuron));
  neuron->nm = params.nm;
  neuron->w = nm_malloc(neuron->nm, sizeof(Value *) * params.nin);
  for (int i = 0; i < params.nin; i++) {
    neuron->w[i] = value_create(
        (float)arc4random_uniform(UINT32_MAX) / UINT32_MAX * 2.0 - 1.0, NULL,
        neuron->nm);
  }
  neuron->b = value_create(
      (float)arc4random_uniform(UINT32_MAX) / UINT32_MAX * 2.0 - 1.0, NULL,
      neuron->nm);
  neuron->nin = params.nin;
  neuron->out = NULL;
  return neuron;
}

Value *neuron_call(Neuron *neuron, Value **x) {
  Value *act = neuron->b;
  for (int i = 0; i < neuron->nin; i++) {
    act = value_add(act, value_multiply(neuron->w[i], x[i], neuron->nm),
                    neuron->nm);
  }
  return value_tanhv(act, neuron->nm);
}

Value **neuron_parameters(Neuron *neuron) {
  Value **params = calloc(neuron->nin + 1, sizeof(Value *));
  for (int i = 0; i < neuron->nin; i++) {
    params[i] = neuron->w[i];
  }
  params[neuron->nin] = neuron->b;
  return params;
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
