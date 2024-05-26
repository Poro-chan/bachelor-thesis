import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from core import FaultInjection
import functions_for_testing as func

#load all images for testing
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

batch_size = 100
testset = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
testset2 = torchvision.datasets.MNIST('../data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

cifartransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #transforms.Resize(299) #for inception_v3
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
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) # densenet121, googlenet, inception_v3, mobilenet_v2, resnet18 (squeezenet1_0, squeezenet1_1) -> excluded
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model.fc = torch.nn.Linear(model.fc.in_features, 10) #for inception_v3
#model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10) # for mobilenet_v2
model.to(device)

learning_rate = 0.7
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

# Define parameters for fault injection
print("\nTraining model on dataset ... ")
#epochs = 1
#func.train(model, device, trainloader, optimizer, epochs)
epochs = 5
func.train(model, device, cifar10trainloader, optimizer, epochs)
#model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=100)
#epochs = 10
#func.train(model, device, cifar100trainloader, optimizer, epochs)
print("Finished training model!\n")

print("Accuracy of the original model:")
func.evaluate(cifar10loader, model, device)
'''
epsilon = 0.0025
while epsilon < 0.55:
    p_custom = func.custom_func(model, batch_size, epsilon, 0, input_shape=[3,224,224], layer_types=[torch.nn.Conv2d], use_cuda=torch.cuda.is_available())
    perturbed_model = p_custom.declare_neuron_fault_injection(function=p_custom.multiply_output)
    perturbed_model.eval()
    print("Multiplication of all values {} for mnist:".format(epsilon))
    func.evaluate(cifar10loader, perturbed_model, device)
    if epsilon > 0.09: epsilon += 0.1
    if 0.01 <= epsilon <= 0.09: epsilon += 0.01
    if epsilon < 0.01: epsilon += 0.0025
    #if epsilon >= 0.01: exit(0)
'''
'''
index = 19
while index < 31 :
        p_custom2 = func.custom_func(model, batch_size, 0, index, input_shape=[3,224,224], layer_types=[torch.nn.Conv2d], use_cuda=torch.cuda.is_available())
        perturbed_model2 = p_custom2.declare_neuron_fault_injection(function=p_custom2.single_bit_flip)
        perturbed_model2.eval()
        print("Accuracy of the model with single bitflip, flipped {} for mnist:".format(index))
        func.evaluate(testloader, perturbed_model2, device)
        index = index + 1

'''
epsilon = 0.01
index = 2
while epsilon < 0.51:
    p_custom = func.custom_func(model, batch_size, epsilon, index, input_shape=[3,224,224], layer_types=[torch.nn.Conv2d], use_cuda=torch.cuda.is_available())
    perturbed_model = p_custom.declare_neuron_fault_injection(function=p_custom.multiply_value_layer)
    perturbed_model.eval()
    print("Multiplication of one value of the {} layer with {} for mnist:".format(index, epsilon))
    func.evaluate(cifar10loader, perturbed_model, device)
    if epsilon > 0.09: epsilon += 0.1
    if 0.01 <= epsilon <= 0.09: epsilon += 0.01
    #if epsilon < 0.01: epsilon += 0.0025

