#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "memory.h"
#include "value.h"

typedef struct MLP {
  Layer **layers;
  int nlayers;
  int outputCount;
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

Value *mlp_call(MLP *mlp, Value **x, nm_t *epochNm);
MLP *mlp_create(mlp_params params);
void mlp_update_parameters(MLP *mlp, double learningRate);
void mlp_zero_grad(MLP *mlp);
void mlp_print(MLP *mlp);

#endif
