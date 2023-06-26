#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "value.h"

typedef struct MLP {
  Layer **layers;
  int nlayers;
  Value ***layerOuts;
} MLP;

typedef struct mlp_params {
  int nin;
  int *nouts;
  int nlayers;
} mlp_params;

MLP *mlp_create(mlp_params params);
ValueList *mlp_call(MLP *mlp, Value **x);
Value *mlp_call_v(MLP *mlp, Value **x);
void mlp_update_graph(MLP *mlp);
Value **mlp_parameters(MLP *mlp);
int mlp_nparams(MLP *mlp);
void mlp_print(MLP *mlp);
void mlp_free(MLP *mlp);

#endif
