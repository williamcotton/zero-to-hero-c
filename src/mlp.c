#include "mlp.h"
#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>

MLP *mlp_create(mlp_params params) {
  MLP *mlp = malloc(sizeof(MLP));
  mlp->layers = malloc(sizeof(Layer *) * params.nlayers);
  for (int i = 0; i < params.nlayers; i++) {
    mlp->layers[i] = layer_create(i == 0 ? params.nin : params.nouts[i - 1],
                                  params.nouts[i], i);
  }
  mlp->nlayers = params.nlayers;
  mlp->layerOuts = malloc(sizeof(Value **) * params.nlayers);
  return mlp;
}

void mlp_print(MLP *mlp) {
  for (int i = 0; i < mlp->nlayers; i++) {
    printf("\nLayer %d: nin=%d nout=%d\n=============================\n", i,
           mlp->layers[i]->nin, mlp->layers[i]->nout);
    for (int j = 0; j < mlp->layers[i]->nout; j++) {
      neuron_print(mlp->layers[i]->neurons[j]);
    }
    printf("-----------------------------\n\n");
  }
}

Value *mlp_call(MLP *mlp, Value **x) {
  if (mlp->nlayers == 0) {
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
  Value *out = NULL;
  for (int i = 0; i < mlp->nlayers; i++) {
    // printf("Layer %d\n", i);
    Value **outs = malloc(sizeof(Value *) * mlp->layers[i]->nout);
    x = layer_call(mlp->layers[i], x, outs);
    mlp->layerOuts[i] = x;
    if (i == mlp->nlayers - 1) {
      out = outs[0];
    }
  }
  Value *outCopy = malloc(sizeof(Value));
  *outCopy = *out;

  for (int i = 0; i < mlp->nlayers; i++) {
    free(mlp->layerOuts[i]);
  }
  if (outCopy)
    outCopy->label = "out";
  return outCopy;
}

int mlp_nparams(MLP *mlp) {
  int total = 0;
  for (int i = 0; i < mlp->nlayers; i++) {
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
  for (int i = 0; i < mlp->nlayers; i++) {
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
  for (int i = 0; i < mlp->nlayers; i++) {
    layer_free(mlp->layers[i]);
  }
  free(mlp->layers);
  free(mlp->layerOuts);
  free(mlp);
}