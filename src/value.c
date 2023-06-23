#include "value.h"
#include <math.h>
#include <stdio.h>

float tanh_float(float x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }

Value add(Value v1, Value v2) {
  Value result = {
      .data = v1.data + v2.data,
      .operation = "+",
      .children = {&v1, &v2},
  };
  return result;
}

Value multiply(Value v1, Value v2) {
  Value result = {
      .data = v1.data * v2.data,
      .operation = "*",
      .children = {&v1, &v2},
  };
  return result;
}

Value initValue(float data, char *label) {
  Value result = {.data = data, .label = label, .grad = 0.0};
  return result;
}

Value tanhv(Value v) {
  Value result = {
      .data = tanh_float(v.data), .operation = "tanh", .children = {&v}};
  return result;
}

void printValue(Value v) {
  printf("Label: %s\n", v.label);
  printf("Data: %f\n", v.data);
  printf("Grad: %f\n", v.grad);
  printf("Operation: %s\n", v.operation);
}
