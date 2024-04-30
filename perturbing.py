import struct
import random
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
# from core import FaultInjection
from pytorchfi.core import FaultInjection
import pytorchfi.neuron_error_models as errorn
import pytorchfi.weight_error_models as errorw

#Train the model
def train(model, device, train_loader, optimizer):
    loss_func = nn.CrossEntropyLoss()
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
            print(f'Epoch {1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')


# Evaluate the model
def evaluate(testloader, model):
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
    def single_bit_flip(self, module, input, output):
        output_detached = output.detach().cpu() if output.requires_grad else output.cpu()

        byte_output = output_detached.numpy().view(np.uint8)

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

#load all images for testing
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

batch_size = 100
testset = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
testset2 = torchvision.datasets.MNIST('../data', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(testset2, batch_size=batch_size, shuffle=True)

cifartransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar10dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifartransform)
cifar10loader = torch.utils.data.DataLoader(cifar10dataset, batch_size=batch_size, shuffle=False)
cifar10trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifartransform)
cifar10trainloader = torch.utils.data.DataLoader(cifar10dataset, batch_size=batch_size, shuffle=True)

cifar100dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=cifartransform)
cifar100loader = torch.utils.data.DataLoader(cifar100dataset, batch_size=batch_size, shuffle=False)
cifar100trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=cifartransform)
cifar100trainloader = torch.utils.data.DataLoader(cifar100dataset, batch_size=batch_size, shuffle=True)

# Load pretrained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
model.to(device)

learning_rate = 1.0
reduce_lr_gamma = 0.7
epochs = 5
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

# Define parameters for fault injection
print("\nTraining model on dataset ... ")
train(model, device, trainloader, optimizer)
#for i in range(2):
    #train(model, device, cifar10trainloader, optimizer)
#model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=100)
#train(model, device, cifar100trainloader, optimizer)
print("Finished training model!\n")

print("Accuracy of the original model for mnist:")
evaluate(testloader, model)

e = 2
p_custom2 = custom_func(model, batch_size, e, input_shape=[3,224,224], layer_types=[torch.nn.Conv2d], use_cuda=torch.cuda.is_available())
perturbed_model2 = p_custom2.declare_neuron_fault_injection(function=p_custom2.single_bit_flip)
perturbed_model2.eval()

# epsilon = 0.01
# while epsilon < 0.51:
#     p_custom = custom_func(model, batch_size, epsilon, input_shape=[3,224,224], layer_types=[torch.nn.Conv2d], use_cuda=False)
#     perturbed_model = p_custom.declare_neuron_fi(function=p_custom.multiply_output)
#     perturbed_model.eval()
#     print("Accuracy of the model with multiplication of {} for mnist:".format(epsilon))
#     evaluate(testloader, perturbed_model)
#     if epsilon > 0.09: epsilon += 0.1
#     else: epsilon += 0.01

print("Accuracy of the model with single bitflip, flipped {} for mnist:".format(e))
evaluate(testloader, perturbed_model2)
exit(0)
#for all models
'''if __name__ == '__main__':
    models_names = ['densenet121',
                    'googlenet',
                    'inception_v3',
                    'mobilenet_v2',
                    'resnet18',
                    'squeezenet1_0',
                    'squeezenet1_1',
                    ]

    for model_name in models_names:
        try:
            model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True).eval()
        except RuntimeError:
            print(f"Model {model_name} not available in the hub.")
        except ValueError:
            print(f"Model {model_name} has no pretrained weights.")'''
