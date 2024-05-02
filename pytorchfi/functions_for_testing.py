#Train the model
import torch
import torch.nn as nn
import numpy as np
from pytorchfi.core import FaultInjection

def train(model, device, train_loader, optimizer, epochs):
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
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

#custom function for bitflip and multiply values by 1+epsilon
class custom_func(FaultInjection):
    def __init__(self, model, batch_size, epsilon, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.epsilon = epsilon

    #bitflip
    def single_bit_flip(self, module, input, output) :
        output_detached = output.detach().cpu() if output.requires_grad else output.cpu()

        byte_output = output_detached.numpy().view(np.uint8)
        
        #for epsilon in self.epsilon :
        index = self.epsilon // 8
        bit_in_byte = self.epsilon % 8

        byte_output[index::output.element_size()] ^= 1 << bit_in_byte

        byte_output = byte_output.view(dtype=np.dtype(output_detached.numpy().dtype))

        output[:] = torch.from_numpy(byte_output).to(output.dtype)

        self.update_layer()
        if self.current_layer >= self.get_total_layers():
            self.reset_current_layer()
    
    #mutiply by a factor
    def multiply_output(self, module, input, output) :
        output[:] = output * (1+self.epsilon)

        self.update_layer()
        if self.current_layer >= self.get_total_layers():
            self.reset_current_layer()
