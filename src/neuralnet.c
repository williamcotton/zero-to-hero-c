#include "neuralnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Neuron *neuron_create(int nin) {
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
  return out;
}

void neuron_free(Neuron *neuron) {
  for (int i = 0; i < neuron->nin; i++) {
    value_free(neuron->w[i]);
  }
  free(neuron->w);
  value_free(neuron->b);
  value_list_free(neuron->out);
  free(neuron);
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

Layer *layer_create(int nin, int nout) {
  Layer *layer = malloc(sizeof(Layer));
  layer->neurons = malloc(sizeof(Neuron *) * nout);
  for (int i = 0; i < nout; i++) {
    layer->neurons[i] = neuron_create(nin);
  }
  layer->nin = nin;
  layer->nout = nout;
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
  Value **params = calloc(layer_nparams(layer), sizeof(Value *));
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

void mlp_print(MLP *mlp) {
  for (int i = 0; i < mlp->n; i++) {
    printf("\nLayer %d: nin=%d nout=%d\n=============================\n", i,
           mlp->layers[i]->nin, mlp->layers[i]->nout);
    for (int j = 0; j < mlp->layers[i]->nout; j++) {
      neuron_print(mlp->layers[i]->neurons[j]);
    }
    printf("-----------------------------\n\n");
  }
}

void value_list_free(ValueList *list) {
  ValueList *next;
  while (list) {
    next = list->next;
    free(list->value);
    free(list);
    list = next;
  }
}

ValueList *value_list_append(ValueList *list, Value *value) {
  ValueList *new_node = malloc(sizeof(ValueList));
  new_node->value = value;
  new_node->next = NULL;

  if (list == NULL) {
    return new_node;
  } else {
    ValueList *current = list;
    while (current->next != NULL) {
      current = current->next;
    }
    current->next = new_node;
    return list;
  }
}

ValueList *mlp_call(MLP *mlp, Value **x) {
  if (mlp->n == 0) {
    printf("Error: MLP has no layers\n");
    return NULL;
  }
  if (x == NULL) {
    printf("Error: Input is NULL\n");
    return NULL;
  }
  if (x[0] == NULL) {
    printf("Error: Input is NULL\n");
    return NULL;
  }
  ValueList *first = NULL, *current = NULL;
  for (int i = 0; i < mlp->n; i++) {
    Value **outs = malloc(sizeof(Value *) * mlp->layers[i]->nout);
    x = layer_call(mlp->layers[i], x, outs);
    for (int j = 0; j < mlp->layers[i]->nout; j++) {
      current = value_list_append(current, outs[j]);
      if (first == NULL) {
        first = current;
      }
    }
  }
  return first;
}

void mlp_update_graph(MLP *mlp) {
  double learning_rate = 0.01;
  for (int i = 0; i < mlp->n; i++) {
    for (int j = 0; j < mlp->layers[i]->nout; j++) {
      for (int k = 0; k < mlp->layers[i]->neurons[j]->nin; k++) {
        mlp->layers[i]->neurons[j]->w[k]->data -=
            mlp->layers[i]->neurons[j]->w[k]->grad;
      }
      mlp->layers[i]->neurons[j]->b->data -=
          learning_rate * mlp->layers[i]->neurons[j]->b->grad;
    }
  }
}

int mlp_nparams(MLP *mlp) {
  int total = 0;
  for (int i = 0; i < mlp->n; i++) {
    int layerCount = layer_nparams(mlp->layers[i]);
    total += layerCount;
  }
  return total;
}

Value **mlp_parameters(MLP *mlp) {
  int paramCount = mlp_nparams(mlp);
  if (paramCount == 0) {
    return NULL;
  }
  Value **params = calloc(paramCount, sizeof(Value *));
  int idx = 0;
  for (int i = 0; i < mlp->n; i++) {
    Value **layer_params = layer_parameters(mlp->layers[i]);
    int layerCount = layer_nparams(mlp->layers[i]);
    for (int j = 0; j < layerCount; j++) {
      params[idx++] = layer_params[j];
    }
    free(layer_params);
  }
  return params;
}

void mlp_free(MLP *mlp) {
  for (int i = 0; i < mlp->n; i++) {
    layer_free(mlp->layers[i]);
  }
  free(mlp->layers);
  free(mlp);
}
