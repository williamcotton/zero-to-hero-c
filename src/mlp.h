#ifndef MLP_H
#define MLP_H

#include "layer.h"
#include "value.h"

typedef struct MLP {
  Layer **layers;
  int n;
} MLP;

MLP *mlp_create(int nin, int *nouts, int n);
ValueList *mlp_call(MLP *mlp, Value **x);
void mlp_update_graph(MLP *mlp);
Value **mlp_parameters(MLP *mlp);
int mlp_nparams(MLP *mlp);
void mlp_print(MLP *mlp);
void mlp_free(MLP *mlp);

#endif
