# Goalkeeper Throw Prediction

This project focuses on predicting whether a goalkeeper's throw will reach a teammate or not using a deep neural network implemented in Keras. The main challenge is to prevent overfitting, as the model has a large number of parameters compared to the limited amount of training data.

To address this, we **compare different regularization techniques** to improve the model's generalization ability and prevent overfitting. Specifically, we evaluate the following approaches:

| Technique               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **No Regularization**   | A deep neural network with no regularization, prone to overfitting.         |
| **Dropout**             | Adding dropout layers (with a rate of 0.6) to randomly deactivate neurons during training. |
| **L1L2 Regularization** | Applying L1 and L2 regularization to the kernel weights of each layer.      |


## Dataset

The dataset contains information about the goalkeeper's throws and the positions of the players on the field. 
## Model Training

### Model without Regularization

We first define a deep neural network model without any regularization techniques. The model consists of:

- **Input Layer**: 2 input features (positions of the throws).
- **Hidden Layers**: Three hidden layers with 5000 neurons each, using ReLU activation.
- **Output Layer**: A single neuron with a sigmoid activation function for binary classification.

The model is compiled using the Adam optimizer and binary cross-entropy loss. It is trained for 500 epochs with a batch size of 32.

### Regularization: Dropout Method

To prevent overfitting, we introduce dropout layers between the hidden layers. The dropout rate is set to 0.6. The model architecture is as follows:

- **Input Layer**: 2 input features.
- **Hidden Layers**: Three hidden layers with 5000 neurons each, using ReLU activation, followed by dropout layers with a rate of 0.6.
- **Output Layer**: A single neuron with a sigmoid activation function.

### Regularization: L1L2 Method

We also experiment with L1L2 regularization for the kernel weights of each layer. The L1 coefficient is set to 3e-5, and the L2 coefficient is set to 3e-4. The model architecture is as follows:

- **Input Layer**: 2 input features.
- **Hidden Layers**: Three hidden layers with 5000 neurons each, using ReLU activation and L1L2 regularization.
- **Output Layer**: A single neuron with a sigmoid activation function and L1L2 regularization.

## Model Performance

The performance of each model is evaluated based on training and validation loss and accuracy. The decision boundaries for each model are also visualized to understand how well the model generalizes to new data.

### Visualization

- **Training and Validation Loss/Accuracy**: Plots showing the training and validation loss and accuracy over epochs.
- **Decision Boundaries**: Visualizations of the decision boundaries for each model, along with the training data points.

## Requirements

To run this project, you need the following Python libraries:
- numpy
- keras
- matplotlib
- scikit-learn
