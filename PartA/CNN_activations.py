import torch.nn.functional as F
class Activation_Function:
    def activation_Function(self,activation_function):
        if activation_function=='relu':
            return F.relu
        if activation_function=='gelu':
            return F.gelu
        if activation_function=='selu':
            return F.selu
        if activation_function=='elu':
            return F.elu
