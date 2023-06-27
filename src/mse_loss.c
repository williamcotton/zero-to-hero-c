#include "mse_loss.h"

Value *mse_loss_call(Value *mseLoss, Value *ypred, Value *ys, nm_t *epochNm) {
  Value *diff = value_subtract(ypred, ys, epochNm);
  Value *loss = value_power(diff, 2, epochNm);
  Value *add = value_add(mseLoss, loss, epochNm);

  return add;
}

Value *mse_loss_create(nm_t *epochNm) {
  Value *mseLoss = value_create(0.0, "mse_loss", epochNm);
  return mseLoss;
}
