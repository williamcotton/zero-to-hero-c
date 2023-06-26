#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>

Neuron *neuron_create(int nin, UNUSED int layer_id, UNUSED int neuron_id) {
  Neuron *neuron = malloc(sizeof(Neuron));
  neuron->w = malloc(sizeof(Value *) * nin);
  for (int i = 0; i < nin; i++) {
    neuron->w[i] = value_create(
        (float)arc4random_uniform(UINT32_MAX) / UINT32_MAX * 2.0 - 1.0, NULL);
  }
  neuron->b = value_create(
      (float)arc4random_uniform(UINT32_MAX) / UINT32_MAX * 2.0 - 1.0, NULL);
  neuron->nin = nin;
  neuron->out = NULL;
  return neuron;
}

Value *neuron_call(Neuron *neuron, Value **x) {
  Value *act = neuron->b;
  for (int i = 0; i < neuron->nin; i++) {
    Value *mul = value_multiply(neuron->w[i], x[i]);
    Value *add = value_add(act, mul);

    neuron->out = value_list_append(neuron->out, mul);
    neuron->out = value_list_append(neuron->out, add);

    act = add;
  }
  Value *out = value_tanhv(act);
  neuron->out = value_list_append(neuron->out, out);

  return out;
}

void neuron_free(Neuron *neuron) {
  for (int i = 0; i < neuron->nin; i++) {
    value_free(neuron->w[i]);
  }
  free(neuron->w);
  value_free(neuron->b);
  neuron_free_value_list(neuron);
  free(neuron);
}

void neuron_free_value_list(Neuron *neuron) {
  ValueList *node = neuron->out;
  while (node != NULL) {
    ValueList *next = node->next;
    value_free(node->value);
    free(node);
    node = next;
  }
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
