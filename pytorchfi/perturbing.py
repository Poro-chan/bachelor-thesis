import torch
import torchvision
import torchvision.transforms as transforms
from core import FaultInjection
import functions_for_testing as func

#load all images for testing
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.Resize((299,299)), #for inception_v3
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

batch_size = 100
mnisttestset = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=transform)
mnistloader = torch.utils.data.DataLoader(mnisttestset, batch_size=batch_size, shuffle=False)
mnisttrainset = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transform)
mnisttrainloader = torch.utils.data.DataLoader(mnisttrainset, batch_size=batch_size, shuffle=True)

cifartransform = transforms.Compose([
        #transforms.Resize((299,299)), #for inception_v3
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
cifar10dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifartransform)
cifar10loader = torch.utils.data.DataLoader(cifar10dataset, batch_size=batch_size, shuffle=False)
cifar10trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifartransform)
cifar10trainloader = torch.utils.data.DataLoader(cifar10trainset, batch_size=batch_size, shuffle=True)

cifar100dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=cifartransform)
cifar100loader = torch.utils.data.DataLoader(cifar100dataset, batch_size=batch_size, shuffle=False)
cifar100trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=cifartransform)
cifar100trainloader = torch.utils.data.DataLoader(cifar100trainset, batch_size=batch_size, shuffle=True)

# Load pretrained model -> specify network in load function
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True) # densenet121, googlenet, inception_v3, mobilenet_v2, resnet18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model.fc = nn.Linear(model.fc.in_features, 10) #for inception_v3
model = model.to(device)

#train model -> uncomment epoch and train function for different datasets
learning_rate = 0.7
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
print("\nTraining model on dataset ... ")

#training with MNIST
epochs = 1
#model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10) # for mobilenet_v2
func.train(model, device, mnisttrainloader, optimizer, epochs)

#training with CIFAR10
#epochs = 5
#model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10) # for mobilenet_v2
#func.train(model, device, cifar10trainloader, optimizer, epochs)

#training with CIFAR100
#epochs = 9
#model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=100) #for mobilenet_v2
#func.train(model, device, cifar100trainloader, optimizer, epochs)

print("Finished training model!\n")

#specify used dataset to use in each evaluation: mnistloader, cifar10loader, cifar100loader
dataset = cifar10loader
print("Accuracy of the original model:")
func.evaluate(dataset, model, device)

#evaluation functions: change function in declare_neuron_fault_injection or the function for the perturbed model to change method
def evaluate_multiplication_values_neurons() :
        epsilon = 0.0025
        index = 0
        while epsilon < 0.55:
                p_custom = func.custom_func(model, batch_size, epsilon, index, input_shape=[3,224,224], layer_types=[torch.nn.Conv2d], use_cuda=torch.cuda.is_available())
                perturbed_model = p_custom.declare_neuron_fault_injection(function=p_custom.multiply_value) #or multiply_value_layer or multiply_one_value_layer
                perturbed_model.eval()
                print("Accuracy of the model with multiplication of neuron values {}:".format(epsilon))
                func.evaluate(dataset, perturbed_model, device)
                if epsilon > 0.09: epsilon += 0.1
                if 0.01 <= epsilon <= 0.09: epsilon += 0.03
                if epsilon < 0.01: epsilon += 0.0025

def evaluate_bitflip_values_neurons() :
        index = 19
        epsilon = 0
        while index < 31 :
                p_custom = func.custom_func(model, batch_size, epsilon, index, input_shape=[3,224,224], layer_types=[torch.nn.Conv2d], use_cuda=torch.cuda.is_available())
                perturbed_model = p_custom.declare_neuron_fault_injection(function=p_custom.single_bit_flip) #or single_bit_flip_layer to flip bits in one layer
                perturbed_model.eval()
                print("Accuracy of the model with single neuron bitflip, flipped {}:".format(index))
                func.evaluate(dataset, perturbed_model, device)
                index = index + 1

def evaluate_multiplication_values_weights() :
        epsilon = 0.0025
        while epsilon < 0.55:
                pfi = FaultInjection(model, batch_size, input_shape=[3,224,224], layer_types=[torch.nn.Conv2d], use_cuda=torch.cuda.is_available())
                perturbed_model = func.perturb_weights(pfi, epsilon) #or perturb_each_layer_weight or perturb_one_layer_weight
                perturbed_model.eval()
                print("Accuracy of the model with multiplication of weight values {}:".format(epsilon))
                func.evaluate(dataset, perturbed_model, device)
                if epsilon > 0.09: epsilon += 0.1
                if 0.01 <= epsilon <= 0.09: epsilon += 0.03
                if epsilon < 0.01: epsilon += 0.0025

def evaluate_bitflip_values_weights() :
        index = 19
        while index < 31 :
                pfi = FaultInjection(model, batch_size, input_shape=[3,224,224], layer_types=[torch.nn.Conv2d], use_cuda=torch.cuda.is_available())
                perturbed_model = func.perturb_weights(pfi, index, func.flip_single_bit) #or perturb_each_layer_weight or perturb_one_layer_weight
                perturbed_model.eval()
                print("Accuracy of the model with single weight bitflip, flipped {}:".format(index))
                func.evaluate(dataset, perturbed_model, device)
                index = index + 1

#evaluation of perturbed models with different methods
evaluate_multiplication_values_neurons()
evaluate_multiplication_values_weights()
evaluate_bitflip_values_neurons()
evaluate_bitflip_values_weights()