# Zero to Deep Learning

This project contains code for a neural network implementation in C. It includes functions for plotting mathematical functions, as well as implementations of neural network components such as neurons, layers, and multi-layer perceptrons (MLPs).

## Dependencies

This project requires the following dependencies:

- `gnuplot`: for plotting mathematical functions
- `leaks`: for detecting memory leaks (optional)

## Building

To build the project, run the following command:

```
make clean && make
```

This will generate an executable file named `main` in the `build` directory.

## Usage

To run the program, simply execute the `main` executable:

```
make run
```

This will run several examples of neural network components, including plotting mathematical functions, creating and calling neurons, layers, and MLPs, and training an MLP.

## Example

### Setting up the input data

This code block sets up the necessary data for machine learning. It first creates an instance of the nm_t object which provides an interface to use memory in a block-wise way. This instance is allocated a block of memory sized `ONE_K * 16` units.

After that, it creates and initializes the input vectors (`xs`) and the corresponding outputs (`ys`). The input vectors are two-dimensional arrays, while the output is a single-dimensional array. The `nm_malloc` function is used to allocate memory from the previously created nm_t instance. Then, `value_create_vector` is used to initialize each vector with the provided data.

```c
nm_t *nm = nm_create(ONE_K * 16);

int outputCount = 4;

Vector **xs = nm_malloc(nm, sizeof(Vector *) * outputCount);
double xs_data[][3] = {
    {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
for (int i = 0; i < outputCount; i++) {
  xs[i] = value_create_vector(xs_data[i], 3, nm);
}

Vector *ys =
    value_create_vector((double[]){1.0, -1.0, -1.0, 1.0}, outputCount, nm);
```

### Creating the model

This code block creates an instance of a Multilayer Perceptron (MLP), a type of artificial neural network. The model has 3 input neurons, 2 hidden layers each with 4 neurons, and 1 output neuron. This is accomplished by using `mlp_create`, which takes an mlp_params object as input. The mlp_params object has properties `nin`, `nouts`, `nlayers`, and `nm`, which define the number of input neurons, the number of output neurons for each layer, the total number of layers, and the nm_t object for memory management, respectively.

```c
MLP *mlp = mlp_create((mlp_params){
    .nin = 3,
    .nouts = (int[]){4, 4, 1},
    .nlayers = 3,
    .nm = nm,
});
```

### Training the model

This code block trains the MLP using the Mean Squared Error (MSE) loss function and the backpropagation algorithm for `epochsCount` number of iterations (epochs).

For each epoch, it first creates a new `nm_t *epochNm` instance to manage memory during that epoch. Then it creates an MSE loss function instance using `mse_loss_create`.

In the forward pass, it computes the prediction of the MLP for each input vector and calculates the MSE loss between the predicted and actual outputs.

In the backward pass, it calls `value_backpropagate` to propagate the loss back through the network and compute the gradients of the model parameters with respect to the loss.

Finally, it updates the model parameters in the direction that minimizes the loss using the learning rate and the calculated gradients. The learning rate is a hyperparameter that determines the step size at each iteration while moving toward a minimum of a loss function. It then frees up the memory allocated to `epochNm`.

```c
int epochsCount = 30;
double learningRate = 0.05;

for (int epoch = 0; epoch < epochsCount; epoch++) {
  // forward pass
  nm_t *epochNm = nm_create(ONE_K * 48);

  Value *mseLoss = mse_loss_create(epochNm);
  for (int i = 0; i < outputCount; i++) {
    Value *ypred = mlp_call(mlp, xs[i]->values, epochNm);
    mseLoss = mse_loss_call(mseLoss, ypred, ys->values[i], epochNm);
    if (epoch == epochsCount - 1) {
      printf("ypred[%d]: %.15lf\n", i, ypred->data);
    }
  }

  // backward pass
  mlp_zero_grad(mlp);
  value_backpropagate(mseLoss, epochNm);

  // update parameters
  mlp_update_parameters(mlp, learningRate);

  printf("%d %.15lf\n", epoch, mseLoss->data);

  nm_free(epochNm);
}
```

### Freeing up the memory

After the training process, the code block releases the memory that was allocated to the `nm` object. This is important for preventing memory leaks which could potentially exhaust the system's memory.

```c
nm_free(nm);
```
