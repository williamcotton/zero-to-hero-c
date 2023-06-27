#include "value.h"
#include "hashset.h"
#include "memory.h"
#include "topographical_sort.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void value_add_backward(Value *v) {
  v->children[0]->grad += 1.0 * v->grad;
  v->children[1]->grad += 1.0 * v->grad;
}

Value *value_add(Value *v1, Value *v2, nm_t *nm) {
  Value *result = nm_malloc(nm, sizeof(Value));
  result->label = NULL;
  result->data = v1->data + v2->data;
  result->operation = "+";
  result->num_children = 2;
  result->children = nm_malloc(nm, sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->children[1] = v2;
  result->grad = 0.0;
  result->backward = value_add_backward;
  return result;
}

static void value_multiply_backward(Value *v) {
  v->children[0]->grad += v->children[1]->data * v->grad;
  v->children[1]->grad += v->children[0]->data * v->grad;
}

Value *value_multiply(Value *v1, Value *v2, nm_t *nm) {
  Value *result = nm_malloc(nm, sizeof(Value));
  result->label = NULL;
  result->data = v1->data * v2->data;
  result->operation = "*";
  result->num_children = 2;
  result->children = nm_malloc(nm, sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->children[1] = v2;
  result->grad = 0.0;
  result->backward = value_multiply_backward;
  return result;
}

static void value_power_backward(Value *v) {
  v->children[0]->grad +=
      v->v2 * pow(v->children[0]->data, v->v2 - 1) * v->grad;
}

Value *value_power(Value *v1, double v2, nm_t *nm) {
  Value *result = nm_malloc(nm, sizeof(Value));
  result->label = NULL;
  result->data = pow(v1->data, v2);
  result->v2 = v2;
  result->operation = "^";
  result->num_children = 1;
  result->children = nm_malloc(nm, sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->grad = 0.0;
  result->backward = value_power_backward;
  return result;
}

Value *value_divide(Value *v1, Value *v2, nm_t *nm) {
  Value *result = value_multiply(v1, value_power(v2, -1.0, nm), nm);
  return result;
}

Value *value_negate(Value *v, nm_t *nm) {
  Value *result = value_multiply(v, value_create(-1.0, "negate", nm), nm);
  return result;
}

static void value_sub_backward(Value *v) {
  v->children[0]->grad += 1.0 * v->grad;
  v->children[1]->grad += -1.0 * v->grad;
}

Value *value_subtract(Value *v1, Value *v2, nm_t *nm) {
  Value *result = nm_malloc(nm, sizeof(Value));
  result->label = NULL;
  result->data = v1->data - v2->data;
  result->operation = "-";
  result->num_children = 2;
  result->children = nm_malloc(nm, sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->children[1] = v2;
  result->grad = 0.0;
  result->backward = value_sub_backward;
  return result;
}

static double tanh_double(double x) {
  return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

static void value_tanhv_backward(Value *v) {
  v->children[0]->grad +=
      (1 - pow(tanh_double(v->children[0]->data), 2)) * v->grad;
}

Value *value_tanhv(Value *v, nm_t *nm) {
  Value *result = nm_malloc(nm, sizeof(Value));
  result->label = NULL;
  result->data = tanh_double(v->data);
  result->operation = "tanh";
  result->num_children = 1;
  result->children = nm_malloc(nm, sizeof(Value *));
  result->children[0] = v;
  result->grad = 0.0;
  result->backward = value_tanhv_backward;
  return result;
}

static void value_expv_backward(Value *v) {
  v->children[0]->grad += v->data * v->grad;
}

Value *value_expv(Value *v, nm_t *nm) {
  Value *result = nm_malloc(nm, sizeof(Value));
  result->label = NULL;
  result->data = exp(v->data);
  result->operation = "exp";
  result->num_children = 1;
  result->children = nm_malloc(nm, sizeof(Value *));
  result->children[0] = v;
  result->grad = 0.0;
  result->backward = value_expv_backward;
  return result;
}

Value *value_create(double data, char *label, nm_t *nm) {
  Value *result = nm_malloc(nm, sizeof(Value));
  if (result == NULL) {
    printf("Failed to allocate memory for Value\n");
    exit(1);
  }
  result->data = data;
  result->label = label;
  result->operation = NULL;
  result->num_children = 0;
  result->children = NULL;
  result->grad = 0.0;
  return result;
}

Vector *value_create_vector(double *data, int size, nm_t *nm) {
  Vector *result = nm_malloc(nm, sizeof(Vector));
  Value **values = nm_malloc(nm, sizeof(Value *) * size);
  for (int i = 0; i < size; i++) {
    char label[10];
    snprintf(label, 10, "%f", data[i]);
    values[i] = value_create(data[i], label, nm);
  }
  result->values = values;
  result->size = size;
  return result;
}

void value_backpropagate(Value *output, nm_t *epochNm) {
  TopoList *topo = topolist_create(10, epochNm);
  int index = 0;
  HashSet *visited = hashset_create(epochNm);
  topolist_sort(output, topo, &index, visited);

  output->grad = 1.0;
  for (int i = topo->size - 1; i >= 0; i--) {
    Value *v = topo->values[i];
    if (v->operation == NULL) {
      continue;
    }
    v->backward(v);
  }
}

void value_print(Value *v, int depth) {
  for (int i = 0; i < depth; i++)
    printf("\t");
  printf("Value\n");

  for (int i = 0; i < depth; i++)
    printf("\t");
  printf("-----\n");

  for (int i = 0; i < depth; i++)
    printf("\t");
  printf("Label: %s\n", v->label);

  for (int i = 0; i < depth; i++)
    printf("\t");
  printf("Data: %f\n", v->data);

  for (int i = 0; i < depth; i++)
    printf("\t");
  printf("Grad: %f\n", v->grad);

  for (int i = 0; i < depth; i++)
    printf("\t");
  printf("Operation: %s\n", v->operation);

  for (int i = 0; i < v->num_children; i++) {
    printf("\n");
    value_print(v->children[i], depth + 1);
  }
}

void value_print_nested(Value *v, int depth) {
  if (v == NULL) {
    return;
  }

  if (v->label && strcmp(v->label, "out") == 0) {
    printf("Value\n");
    printf("-----\n");
    printf("Label: %s\n", v->label);
    printf("Data: %f\n", v->data);
    printf("Grad: %f\n", v->grad);
    printf("Operation: %s\n", v->operation);
    return;
  }

  printf("Label: %s\n", v->label);
  for (int i = 0; i < v->num_children; i++) {
    value_print_nested(v->children[i], depth--);
  }
}
