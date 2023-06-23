#include "value.h"
#include "hashset.h"
#include "topographical_sort.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float tanh_float(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }

Value *add(Value *v1, Value *v2) {
  Value *result = malloc(sizeof(Value));
  result->data = v1->data + v2->data;
  result->operation = "+";
  result->num_children = 2;
  result->children = malloc(sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->children[1] = v2;
  result->grad = 0.0;
  return result;
}

Value *multiply(Value *v1, Value *v2) {
  Value *result = malloc(sizeof(Value));
  result->data = v1->data * v2->data;
  result->operation = "*";
  result->num_children = 2;
  result->children = malloc(sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->children[1] = v2;
  result->grad = 0.0;
  return result;
}

Value *power(Value *v1, Value *v2) {
  Value *result = malloc(sizeof(Value));
  result->data = pow(v1->data, v2->data);
  result->operation = "^";
  result->num_children = 2;
  result->children = malloc(sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->children[1] = v2;
  result->grad = 0.0;
  return result;
}

Value *division(UNUSED Value *v1, Value *v2) {
  Value *result = power(v2, initValue(-1.0, "-1"));
  return result;
}

Value *expv(Value *v) {
  Value *result = malloc(sizeof(Value));
  result->data = exp(v->data);
  result->operation = "exp";
  result->num_children = 1;
  result->children = malloc(sizeof(Value *));
  result->children[0] = v;
  result->grad = 0.0;
  return result;
}

Value *initValue(float data, char *label) {
  Value *result = malloc(sizeof(Value));
  result->data = data;
  result->label = label;
  result->operation = NULL;
  result->num_children = 0;
  result->children = NULL;
  result->grad = 0.0;
  return result;
}

Value *tanhv(Value *v) {
  Value *result = malloc(sizeof(Value));
  result->data = tanh_float(v->data);
  result->operation = "tanh";
  result->num_children = 1;
  result->children = malloc(sizeof(Value *));
  result->children[0] = v;
  result->grad = 0.0;
  return result;
}

void add_backward(Value *v) {
  v->children[0]->grad += 1.0 * v->grad;
  v->children[1]->grad += 1.0 * v->grad;
}

void multiply_backward(Value *v) {
  v->children[0]->grad += v->children[1]->data * v->grad;
  v->children[1]->grad += v->children[0]->data * v->grad;
}

void expv_backward(Value *v) { v->children[0]->grad += v->data * v->grad; }

void tanhv_backward(Value *v) {
  v->children[0]->grad +=
      (1 - pow(tanh_float(v->children[0]->data), 2)) * v->grad;
}

void power_backward(Value *v) {
  v->children[0]->grad += v->children[1]->data *
                          pow(v->children[0]->data, v->children[1]->data - 1) *
                          v->grad;
  v->children[1]->grad += log(v->children[0]->data) *
                          pow(v->children[0]->data, v->children[1]->data) *
                          v->grad;
}

void backpropagate(Value *v) {
  if (v->operation == NULL) {
    return;
  }
  if (strcmp(v->operation, "+") == 0) {
    add_backward(v);
  } else if (strcmp(v->operation, "*") == 0) {
    multiply_backward(v);
  } else if (strcmp(v->operation, "tanh") == 0) {
    tanhv_backward(v);
  } else if (strcmp(v->operation, "exp") == 0) {
    expv_backward(v);
  } else if (strcmp(v->operation, "^") == 0) {
    power_backward(v);
  }
}

void backpropagateGraph(Value *output) {
  TopoList *topo = topolist_create(10);
  int index = 0;
  HashSet *visited = hashset_create();
  topographicalSort(output, topo, &index, visited);

  output->grad = 1.0;
  for (int i = topo->size - 1; i >= 0; i--) {
    backpropagate(topo->values[i]);
  }
  topolist_free(topo);
}

void printValue(Value *v) {
  printf("Value:\n");
  printf("Label: %s\n", v->label);
  printf("Data: %f\n", v->data);
  printf("Grad: %f\n", v->grad);
  printf("Operation: %s\n", v->operation);
  for (int i = 0; i < v->num_children; i++) {
    printf("Child %d:\n", i);
    printValue(v->children[i]);
  }
}

void freeValue(Value *v) {
  for (int i = 0; i < v->num_children; i++) {
    freeValue(v->children[i]);
  }
  free(v->children);
  free(v);
}
