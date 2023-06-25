#include "value.h"
#include "hashset.h"
#include "topographical_sort.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// void value_divide_backward(Value *v) {
//   v->children[0]->grad += 1.0 / v->children[1]->data * v->grad;
//   v->children[1]->grad += -1.0 * v->children[0]->data /
//                           (v->children[1]->data * v->children[1]->data) *
//                           v->grad;
// }

void value_add_backward(Value *v) {
  v->children[0]->grad += 1.0 * v->grad;
  v->children[1]->grad += 1.0 * v->grad;
}

Value *value_add(Value *v1, Value *v2) {
  Value *result = malloc(sizeof(Value));
  result->label = NULL;
  result->data = v1->data + v2->data;
  result->operation = "+";
  result->num_children = 2;
  result->children = malloc(sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->children[1] = v2;
  result->grad = 0.0;
  result->backward = value_add_backward;
  return result;
}

void value_multiply_backward(Value *v) {
  v->children[0]->grad += v->children[1]->data * v->grad;
  v->children[1]->grad += v->children[0]->data * v->grad;
}

Value *value_multiply(Value *v1, Value *v2) {
  Value *result = malloc(sizeof(Value));
  result->label = NULL;
  result->data = v1->data * v2->data;
  result->operation = "*";
  result->num_children = 2;
  result->children = malloc(sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->children[1] = v2;
  result->grad = 0.0;
  result->backward = value_multiply_backward;
  return result;
}

void value_power_backward(Value *v) {
  v->children[0]->grad +=
      v->v2 * pow(v->children[0]->data, v->v2 - 1) * v->grad;
}

Value *value_power(Value *v1, double v2) {
  Value *result = malloc(sizeof(Value));
  result->label = NULL;
  result->data = pow(v1->data, v2);
  result->v2 = v2;
  result->operation = "^";
  result->num_children = 1;
  result->children = malloc(sizeof(Value *) * result->num_children);
  result->children[0] = v1;
  result->grad = 0.0;
  result->backward = value_power_backward;
  return result;
}

// Value *value_divide(Value *v1, Value *v2) {
//   Value *result = malloc(sizeof(Value));
//   result->label = NULL;
//   result->data = v1->data / v2->data;
//   result->operation = "/";
//   result->num_children = 2;
//   result->children = malloc(sizeof(Value *) * result->num_children);
//   result->children[0] = v1;
//   result->children[1] = v2;
//   result->grad = 0.0;
//   result->backward = value_divide_backward;
//   return result;
// }

Value *value_divide(Value *v1, Value *v2) {
  Value *result = value_multiply(v1, value_power(v2, -1.0));
  return result;
}

Value *value_negate(Value *v) {
  Value *result = value_multiply(v, value_create(-1.0, NULL));
  return result;
}

Value *value_subtract(Value *v1, Value *v2) {
  return value_add(v1, value_negate(v2));
}

double tanh_double(double x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }

// void value_tanhv_backward(Value *v) {
//   v->children[0]->grad += (1 - pow(v->data, 2)) * v->grad;
// }

void value_tanhv_backward(Value *v) {
  v->children[0]->grad +=
      (1 - pow(tanh_double(v->children[0]->data), 2)) * v->grad;
}

Value *value_tanhv(Value *v) {
  Value *result = malloc(sizeof(Value));
  result->label = NULL;
  result->data = tanh_double(v->data);
  result->operation = "tanh";
  result->num_children = 1;
  result->children = malloc(sizeof(Value *));
  result->children[0] = v;
  result->grad = 0.0;
  result->backward = value_tanhv_backward;
  return result;
}

void value_expv_backward(Value *v) {
  v->children[0]->grad += v->data * v->grad;
}

Value *value_expv(Value *v) {
  Value *result = malloc(sizeof(Value));
  result->label = NULL;
  result->data = exp(v->data);
  result->operation = "exp";
  result->num_children = 1;
  result->children = malloc(sizeof(Value *));
  result->children[0] = v;
  result->grad = 0.0;
  result->backward = value_expv_backward;
  return result;
}

// Value *value_sum(Value **values, int num_values) {
//   Value *result = malloc(sizeof(Value));
//   result->label = NULL;
//   result->data = 0.0;
//   result->operation = "sum";
//   result->num_children = num_values;
//   result->children = malloc(sizeof(Value *) * result->num_children);
//   for (int i = 0; i < num_values; i++) {
//     result->data += values[i]->data;
//     result->children[i] = values[i];
//   }
//   result->grad = 0.0;
//   result->backward = value_add_backward;
//   return result;
// }

Value *value_create(double data, char *label) {
  Value *result = malloc(sizeof(Value));
  result->data = data;
  result->label = label;
  result->operation = NULL;
  result->num_children = 0;
  result->children = NULL;
  result->grad = 0.0;
  return result;
}

void value_backpropagate_graph(Value *output) {
  TopoList *topo = topolist_create(10);
  int index = 0;
  HashSet *visited = hashset_create();
  topolist_sort(output, topo, &index, visited);

  output->grad = 1.0;
  for (int i = topo->size - 1; i >= 0; i--) {
    Value *v = topo->values[i];
    if (v->operation == NULL) {
      continue;
    }
    v->backward(v);
  }
  topolist_free(topo);
}

void value_update_graph(Value *output, float learning_rate) {
  TopoList *topo = topolist_create(10);
  int index = 0;
  HashSet *visited = hashset_create();
  topolist_sort(output, topo, &index, visited);

  for (int i = topo->size - 1; i >= 0; i--) {
    Value *v = topo->values[i];
    if (v->operation == NULL) {
      continue;
    }
    v->data -= learning_rate * v->grad;
  }
  topolist_free(topo);
}

void value_zero_grad_graph(Value *output) {
  TopoList *topo = topolist_create(10);
  int index = 0;
  HashSet *visited = hashset_create();
  topolist_sort(output, topo, &index, visited);

  // printf("topo size: %d\n", topo->size);

  for (int i = topo->size - 1; i >= 0; i--) {
    Value *v = topo->values[i];
    if (v->operation == NULL) {
      continue;
    }
    v->grad = 0.0;
  }
  topolist_free(topo);
}

void value_free_graph(Value *output) {
  TopoList *topo = topolist_create(10);
  int index = 0;
  HashSet *visited = hashset_create();
  topolist_sort(output, topo, &index, visited);

  for (int i = topo->size - 1; i >= 0; i--) {
    Value *v = topo->values[i];
    if (v->operation == NULL) {
      continue;
    }
    free(v->children);
    free(v);
  }
  topolist_free(topo);
}

void value_print(Value *v) {
  printf("Value:\n");
  printf("Label: %s\n", v->label);
  printf("Data: %f\n", v->data);
  printf("Grad: %f\n", v->grad);
  printf("Operation: %s\n", v->operation);
  for (int i = 0; i < v->num_children; i++) {
    printf("Child %d:\n", i);
    value_print(v->children[i]);
  }
}

Value *value_copy(Value *dst, Value *src) {
  dst->data = src->data;
  dst->grad = src->grad;
  // dst->label = src->label;
  // dst->operation = src->operation;
  // dst->num_children = src->num_children;
  // dst->children = malloc(sizeof(Value *) * dst->num_children);
  // for (int i = 0; i < dst->num_children; i++) {
  //   dst->children[i] = value_copy(malloc(sizeof(Value)), src->children[i]);
  // }
  // dst->backward = src->backward;
  return dst;
}

void value_free(Value *v) {
  for (int i = 0; i < v->num_children; i++) {
    value_free(v->children[i]);
  }
  free(v->children);
  free(v);
}
