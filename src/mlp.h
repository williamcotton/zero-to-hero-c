#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "memory.h"
#include "value.h"

typedef struct MLP {
  Layer **layers;
  int nlayers;
  Value ***layerOuts;
  int outputCount;
  ValueList *losses;
  Value **params;
  int paramsCount;
  nm_t *nm;
} MLP;

typedef struct mlp_params {
  int nin;
  int *nouts;
  int nlayers;
  nm_t *nm;
} mlp_params;

MLP *mlp_create(mlp_params params);
Value *mlp_call(MLP *mlp, Value **x);
Value **mlp_parameters(MLP *mlp);
void mlp_update_parameters(MLP *mlp, double learningRate);
void mlp_zero_grad(MLP *mlp);
int mlp_nparams(MLP *mlp);
void mlp_print(MLP *mlp);
void mlp_free(MLP *mlp);
void mlp_add_loss_function(MLP *mlp, Value *loss);
Value *mlp_compute_mse_loss(MLP *mlp, Value *mseLoss, Value *ypred, Value *ys);
Value *mlp_create_mse_loss(MLP *mlp);
void mlp_free_loss_functions(MLP *mlp);

#endif
