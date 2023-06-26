#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "value.h"

typedef struct MLP {
  Layer **layers;
  int nlayers;
  Value ***layerOuts;
  int outputCount;
} MLP;

typedef struct mlp_params {
  int nin;
  int *nouts;
  int nlayers;
} mlp_params;

MLP *mlp_create(mlp_params params);
Value *mlp_call(MLP *mlp, Value **x);
Value **mlp_parameters(MLP *mlp);
int mlp_nparams(MLP *mlp);
void mlp_print(MLP *mlp);
void mlp_free(MLP *mlp);
void mlp_free_layer_outs(MLP *mlp);

#endif
