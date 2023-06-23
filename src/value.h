#ifndef VALUE_H
#define VALUE_H

#define MAX_CHILDREN 1024 * 10

#define UNUSED __attribute__((unused))

typedef struct Value {
  double data;
  char *label;
  char *operation;
  struct Value **children;
  int num_children;
  float grad;
} Value;

Value *add(Value *v1, Value *v2);
Value *multiply(Value *v1, Value *v2);
Value *expv(Value *v);
Value *tanhv(Value *v);
Value *initValue(float data, char *label);
void printValue(Value *v);
void freeValue(Value *v);
void backpropagate(Value *v);
void backpropagateGraph(Value *output);

#endif
