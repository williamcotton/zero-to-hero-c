#include "value.h"
#include <math.h>
#include <stdio.h>

static void addBackward(struct Value *self, struct Value *other,
                        struct Value *out) {
  self->grad += 1.0 * out->grad;
  other->grad += 1.0 * out->grad;
}

Value add(Value v1, Value v2) {
  Value result = {.data = v1.data + v2.data,
                  .operation = "+",
                  .children = {&v1, &v2},
                  .backward = &addBackward};
  return result;
}

static void multiplyBackward(struct Value *self, struct Value *other,
                             struct Value *out) {
  self->grad += other->data * out->grad;
  other->grad += self->data * out->grad;
}

Value multiply(Value v1, Value v2) {
  Value result = {.data = v1.data * v2.data,
                  .operation = "*",
                  .children = {&v1, &v2},
                  .backward = &multiplyBackward};
  return result;
}

float tanh_float(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }

static void tanhBackward(struct Value *self, struct Value *other,
                         struct Value *out) {
  self->grad += (1 - pow(tanh_float(self->data), 2)) * out->grad;
}

Value tanhv(Value v) {
  Value result = {.data = tanh_float(v.data),
                  .operation = "tanh",
                  .children = {&v},
                  .backward = &tanhBackward};
  return result;
}

Value initValue(float data, char *label) {
  Value result = {.data = data, .label = label, .grad = 0.0};
  return result;
}

void printValue(Value v) {
  printf("Label: %s\n", v.label);
  printf("Data: %f\n", v.data);
  printf("Grad: %f\n", v.grad);
  printf("Operation: %s\n", v.operation);
}
