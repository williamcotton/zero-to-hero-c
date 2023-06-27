#include "mlp.h"
#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>

static int mlp_nparams(MLP *mlp) {
  int total = 0;
  for (int i = 0; i < mlp->nlayers; i++) {
    int layerCount = layer_nparams(mlp->layers[i]);
    total += layerCount;
  }
  return total;
}

static Value **mlp_parameters(MLP *mlp) {
  Value **params = nm_calloc(mlp->nm, mlp->paramsCount, sizeof(Value *));
  int idx = 0;
  for (int i = 0; i < mlp->nlayers; i++) {
    Value **layer_params = layer_parameters(mlp->layers[i]);
    int layerCount = layer_nparams(mlp->layers[i]);
    for (int j = 0; j < layerCount; j++) {
      params[idx++] = layer_params[j];
    }
  }
  return params;
}

MLP *mlp_create(mlp_params params) {
  MLP *mlp = nm_malloc(params.nm, sizeof(MLP));
  mlp->nm = params.nm;
  mlp->layers = nm_malloc(mlp->nm, sizeof(Layer *) * params.nlayers);
  for (int i = 0; i < params.nlayers; i++) {
    mlp->layers[i] = layer_create((layer_params){
        .nin = i == 0 ? params.nin : params.nouts[i - 1],
        .nout = params.nouts[i],
        .layer_id = i,
        .nm = mlp->nm,
    });
  }
  mlp->nlayers = params.nlayers;
  mlp->paramsCount = mlp_nparams(mlp);
  mlp->params = mlp_parameters(mlp);
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

Value *mlp_call(MLP *mlp, Value **x, nm_t *epochNm) {
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
    Value **outs = nm_malloc(epochNm, sizeof(Value *) * mlp->layers[i]->nout);
    x = layer_call(mlp->layers[i], x, outs, epochNm);
    if (i == mlp->nlayers - 1) {
      out = x[0];
    }
  }
  return out;
}

void mlp_update_parameters(MLP *mlp, double learningRate) {
  for (int i = 0; i < mlp->paramsCount; i++) {
    mlp->params[i]->data += -learningRate * mlp->params[i]->grad;
  }
}

void mlp_zero_grad(MLP *mlp) {
  for (int i = 0; i < mlp->paramsCount; i++) {
    mlp->params[i]->grad = 0.0;
  }
}
