#include "mlp.h"
#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>

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
  mlp->layerOuts = nm_malloc(mlp->nm, sizeof(Value **) * params.nlayers);
  mlp->losses = NULL;
  mlp->params = mlp_parameters(mlp);
  mlp->paramsCount = mlp_nparams(mlp);
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
    Value **outs =
        nm_malloc(mlp->epochNm, sizeof(Value *) * mlp->layers[i]->nout);
    x = layer_call(mlp->layers[i], x, outs);
    mlp->layerOuts[i] = x;
    if (i == mlp->nlayers - 1) {
      out = outs[0];
    }
  }
  return out;
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
  Value **params = nm_calloc(mlp->nm, paramCount, sizeof(Value *));
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

void mlp_add_loss_function(MLP *mlp, Value *loss) {
  ValueList *node = nm_malloc(mlp->epochNm, sizeof(ValueList));
  node->value = loss;
  node->next = NULL;
  if (mlp->losses == NULL) {
    mlp->losses = node;
  } else {
    ValueList *curr = mlp->losses;
    while (curr->next != NULL) {
      curr = curr->next;
    }
    curr->next = node;
  }
}

void mlp_free_loss_functions(MLP *mlp) {
  ValueList *curr = mlp->losses;
  while (curr != NULL) {
    ValueList *next = curr->next;
    value_free(curr->value);
    curr = next;
  }
  mlp->losses = NULL;
}

Value *mlp_compute_mse_loss(MLP *mlp, Value *mseLoss, Value *ypred, Value *ys) {
  Value *diff = value_subtract(ypred, ys);
  mlp_add_loss_function(mlp, diff);

  Value *loss = value_power(diff, 2);
  mlp_add_loss_function(mlp, loss);

  Value *add = value_add(mseLoss, loss);
  mlp_add_loss_function(mlp, add);

  return add;
}

Value *mlp_create_mse_loss(MLP *mlp) {
  Value *mseLoss = value_create(0.0, "mse_loss");
  mlp_add_loss_function(mlp, mseLoss);
  return mseLoss;
}

void mlp_free(MLP *mlp) {
  for (int i = 0; i < mlp->nlayers; i++) {
    layer_free(mlp->layers[i]);
  }
  mlp_free_loss_functions(mlp);
}