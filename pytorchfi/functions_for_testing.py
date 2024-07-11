#Train the model
import torch
import torch.nn as nn
import numpy as np
from core import FaultInjection
import random as random

#train the model
def train(model, device, train_loader, optimizer, epochs):
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print("\nEpoch: {}".format(epoch+1))
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            if isinstance(outputs, tuple):
                output, aux_outputs = outputs
            else:
                output = outputs
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')


# Evaluate the model
def evaluate(testloader, model, device):
  dataset_size = len(testloader.dataset)
  correct = 0
  model.eval()
  with torch.no_grad():
      for image, label in testloader:
          image, label = image.to(device), label.to(device)
          output = model(image)
          _, predicted = torch.max(output, 1)
          correct += torch.sum(predicted == label.data).item()

  print('Accuracy: {}/{} ({:.2f}%)\n'.format(
          correct, dataset_size, 100. * correct / dataset_size))

#Weight perturbations
#1+epsilon weight perturbation
def multiply_value_weight(data, location, epsilon) :
    perturbed_data = data[location] * (1+epsilon)
    return perturbed_data

#single bit flip perturbation
def flip_single_bit(data, location, index) :
    shape = data[location].shape
    data_cpu = data[location].detach().cpu()

    int32_data = np.array([data_cpu.item()], dtype=np.float32).view(np.int32)

    bit_mask = 1 << index
    int32_data = int32_data ^ bit_mask

    float_data = torch.from_numpy(int32_data.view(np.float32))
    float_data = float_data.reshape(shape)
    
    if data[location].is_cuda :
        return float_data.to(float_data.device)
    else:
        return float_data

#save all weights of one specified layer for perturbing
def perturb_one_layer_weight(pfi: FaultInjection, epsilon, index, function = multiply_value_weight):
    corrupt_idx = [[], [], [], [], []]
    shape = list(pfi.get_weights_size(index))
    dim_len = len(shape)
    shape.extend([1 for _ in range(4 - len(shape))])
        
    for k in range(shape[0]):
        for dim1 in range(shape[1]):
            for dim2 in range(shape[2]):
                for dim3 in range(shape[3]):
                    idx = [index, k, dim1, dim2, dim3]
                    for i in range(dim_len + 1):
                        corrupt_idx[i].append(idx[i])
    
    def wrapped_function(data, location) :
        return function(data, location, epsilon)
        
    return pfi.declare_weight_fault_injection(
        layer_num=corrupt_idx[0],
        k=corrupt_idx[1],
        dim1=corrupt_idx[2],
        dim2=corrupt_idx[3],
        dim3=corrupt_idx[4],
        function=wrapped_function
    )

#save all weights at the specified index for perturbing
def perturb_each_layer_weight(pfi: FaultInjection, epsilon, index, function = multiply_value_weight):
    corrupt_idx = [[], [], [], [], []]
    for layer_idx in range(pfi.get_total_layers()):
        shape = list(pfi.get_weights_size(layer_idx))
        shape.extend([1 for _ in range(4 - len(shape))])
        dim_len = len(shape)
        
        k = index // (shape[1] * shape[2] * shape[3])
        remaining = index % (shape[1] * shape[2] * shape[3])
        dim1 = remaining // (shape[2] * shape[3])
        remaining = remaining % (shape[2] * shape[3])
        dim2 = remaining // shape[3]
        dim3 = remaining % shape[3]
        idx = [layer_idx, k, dim1, dim2, dim3]
        for i in range(dim_len + 1):
            corrupt_idx[i].append(idx[i])

    def wrapped_function(data, location) :
        return function(data, location, epsilon)
        
    return pfi.declare_weight_fault_injection(
        layer_num=corrupt_idx[0],
        k=corrupt_idx[1],
        dim1=corrupt_idx[2],
        dim2=corrupt_idx[3],
        dim3=corrupt_idx[4],
        function=wrapped_function
    )

#save all weights for perturbing
def perturb_weights(pfi: FaultInjection, epsilon, function = multiply_value_weight):
    corrupt_idx = [[], [], [], [], []]
    for layer_idx in range(pfi.get_total_layers()):
        shape = list(pfi.get_weights_size(layer_idx))
        dim_len = len(shape)
        shape.extend([1 for _ in range(4 - len(shape))])
        if layer_idx < 5 :
            for k in range(shape[0]):
                for dim1 in range(shape[1]):
                    for dim2 in range(shape[2]):
                        for dim3 in range(shape[3]):
                            idx = [layer_idx, k, dim1, dim2, dim3]
                            for i in range(dim_len + 1):
                                corrupt_idx[i].append(idx[i])
    
    def wrapped_function(data, location) :
        return function(data, location, epsilon)

    return pfi.declare_weight_fault_injection(
        layer_num=corrupt_idx[0],
        k=corrupt_idx[1],
        dim1=corrupt_idx[2],
        dim2=corrupt_idx[3],
        dim3=corrupt_idx[4],
        function=wrapped_function
    )

#custom function for bitflip and multiply values by 1+epsilon
class custom_func(FaultInjection):
    def __init__(self, model, batch_size, epsilon, index, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.epsilon = epsilon
        self.index = index

    #single bitflip in 32-bit representation
    def single_bit_flip(self, module, input, output) :
        shape = output.shape
        output_cpu = output.detach().cpu()
        int32_output = output_cpu.numpy().view(np.int32)

        bit_mask = 1 << self.index
        int32_output = int32_output ^ bit_mask

        float_output = torch.from_numpy(int32_output.view(np.float32))

        # Reshape back to the original shape
        float_output = float_output.reshape(shape)

        if output.is_cuda :
            output.data.copy_(float_output.to(output.device))
        else:
            output.data.copy_(float_output)

        self.update_layer()
        if self.current_layer >= self.get_total_layers():
            self.reset_current_layer()

    #flip only one bit in one layer
    def single_bit_flip_layer(self, module, input, output) :
        if self.current_layer == self.epsilon:
            shape = output.shape
            output_cpu = output.detach().cpu()
            int32_output = output_cpu.numpy().view(np.int32)

            bit_mask = 1 << self.index
            int32_output = int32_output ^ bit_mask

            float_output = torch.from_numpy(int32_output.view(np.float32))
            float_output = float_output.reshape(shape)

            if output.is_cuda :
                output.data.copy_(float_output.to(output.device))
            else:
                output.data.copy_(float_output)

        self.update_layer()
        if self.current_layer >= self.get_total_layers():
            self.reset_current_layer()
    
    #mutiply all output by a factor
    def multiply_output(self, module, input, output) :
        output[:] = output * (1+self.epsilon)

        self.update_layer()
        if self.current_layer >= self.get_total_layers():
            self.reset_current_layer()

    #multiply only one value by a factor
    #see whats more vulnerable test multiple things: eg 1 value per layer fixed/random or all values for one layer 
    def multiply_value(self, module, input, output) :
        output[:] = output * (1+self.epsilon)

        self.update_layer()
        if self.current_layer >= self.get_total_layers():
            self.reset_current_layer()

    def multiply_value_layer(self, module, input, output) :
        if(self.index == self.current_layer) :
            output[:] = output * (1+self.epsilon)

        self.update_layer()
        if self.current_layer >= self.get_total_layers():
            self.reset_current_layer()

    def multiply_one_value_layer(self, module, input, output) :
        output[self.index] = output[self.index] * (1+self.epsilon)

        self.update_layer()
        if self.current_layer >= self.get_total_layers():
            self.reset_current_layer()