#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "memory.h"
#include "mlp.h"
#include "value.h"

Value *mse_loss_call(MLP *mlp, Value *mseLoss, Value *ypred, Value *ys,
                     nm_t *nm);
Value *mse_loss_create(MLP *mlp, nm_t *nm);

#endif
