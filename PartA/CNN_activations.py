import torch.nn.functional as F
class Activation_Function:
    def activation_Function(self,activation_function):   # Check the provided activation function and return the corresponding function from torch.nn.functional
        if activation_function=='relu':     # Rectified Linear Unit (ReLU) activation function
            return F.relu
        if activation_function=='gelu':      # Gaussian Error Linear Unit (GELU) activation function
            return F.gelu
        if activation_function=='selu':       # Scaled Exponential Linear Unit (SELU) activation function
            return F.selu
        if activation_function=='elu':         # Exponential Linear Unit (ELU) activation function
            return F.elu
