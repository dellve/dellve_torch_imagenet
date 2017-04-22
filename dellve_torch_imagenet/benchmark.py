
import dellve

import os
import shutil
import time

import torch
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins as plugins
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

class DataSet(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        if 'size_limit' in kwargs:
            size_limit = kwargs['size_limit']
            del kwargs['size_limit']
        else:
            size_limit = None

        datasets.CIFAR10.__init__(self, *args, **kwargs)

        if size_limit is None:
            self.__size_limit = datasets.CIFAR10.__len__(self)
        else:
            if size_limit > datasets.CIFAR10.__len__(self):
                raise ValueError('CIFAR10 size_limit is too large')
            elif size_limit < 0:
                raise ValueError('CIFAR10 size_limit must be positive')
            self.__size_limit = size_limit

    def __len__(self):
        return self.__size_limit

    def __getitem__(self, index):
        if index >= self.__size_limit:
            raise KeyError()
        else:
            return datasets.CIFAR10.__getitem__(self, index)

class Trainer(trainer.Trainer):

    def __init__(self, benchmark, *args, **kwargs):
        trainer.Trainer.__init__(self, *args, **kwargs)
        # Save DELLve benchmark
        #
        # Note: we need benchmark object in train() method to update progress
        self.benchmark = benchmark

    def run(self, epochs=1):
        # Save number of epochs
        #
        # Note: we need number of epochs in train() method to calculate progress
        #
        self.epochs = epochs

        # Initialize progress
        self.benchmark.progress = 0

        # Run trainer
        trainer.Trainer.run(self, epochs)

        # Finilize progress
        self.benchmark.progress = 100

    def train(self):
        for i, data in enumerate(self.dataset, self.iterations + 1):
            batch_input, batch_target = data
            self.call_plugins('batch', i, batch_input, batch_target)
            input_var = torch.autograd.Variable(batch_input)
            target_var = torch.autograd.Variable(batch_target).squeeze()

            plugin_data = [None, None]

            def closure():
                batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var)
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', i, batch_input, batch_target,
                              *plugin_data)
            self.call_plugins('update', i, self.model)

            # Update benchmark progress
            self.benchmark.progress = \
                int(100 * float(i) / (len(self.dataset) * self.epochs))

        self.iterations = i

class TorchImagenetBenchmark(dellve.Benchmark):

    name = 'TorchImagenetBenchmark'

    config = dellve.BenchmarkConfig([])

    def benchmark(self, model_name,
        device_id=1,
        num_workers=0,
        num_epochs=3,
        batch_size=256,
        train_size_limit=5000,
        valid_size_limit=1000,
        learning_rate=0.1,
        sgd_momentum=0.9,
        sgd_weight_decay=1e-4,
        print_freq=1):

        torch.cuda.set_device(device_id)

        model = getattr(models, model_name)().cuda()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        target_transform = \
            transforms.Lambda(lambda t: torch.cuda.LongTensor([t]))

        train_data_loader = torch.utils.data.DataLoader(
            DataSet('data_cifar10', size_limit=train_size_limit, train=True,
                transform=transforms.Compose([
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                    transforms.Lambda(lambda t: t.cuda(device_id))
                ]),
                target_transform=target_transform,
                download=True
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)

        val_data_loader = torch.utils.data.DataLoader(
            DataSet('data_cifar10', size_limit=valid_size_limit, train=False,
                transform=transforms.Compose([
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                    transforms.Lambda(lambda t: t.cuda(device_id))
                ]),
                target_transform=target_transform,
                download=True
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)

        criterion = nn.CrossEntropyLoss().cuda(device_id)

        optimizer = torch.optim.SGD(model.parameters(),
                                    learning_rate,
                                    momentum=sgd_momentum,
                                    weight_decay=sgd_weight_decay)

        t = Trainer(self, model, criterion, optimizer, train_data_loader)

        print 'Start training...'

        t.run(num_epochs)

    def run_benchmark(self, model_name):
        self.benchmark(model_name)


class AlexnetBenchmark(TorchImagenetBenchmark):

    name = 'AlexnetBenchmark'

    def routine(self):
        self.run_benchmark('alexnet')

# Note: all of the benchmarks bellow result in
#       CUDA out of memory errors.
#
#class ResnetBenchmark(TorchImagenetBenchmark):
#
#    name = 'ResnetBenchmark'
#
#    def run(self):
#        self.run_benchmark('resnet')
#
#class Resnet101Benchmark(TorchImagenetBenchmark):
#
#    name = 'Resnet101Benchmark'
#
#    def run(self):
#        self.run_benchmark('resnet101')
#
#class Resnet152Benchmark(TorchImagenetBenchmark):
#
#    name = 'Resnet152Benchmark'
#
#    def run(self):
#        self.run_benchmark('resnet152')
#
#class Resnet18Benchmark(TorchImagenetBenchmark):
#
#    name = 'Resnet18Benchmark'
#
#    def run(self):
#        self.run_benchmark('resnet18')
#
#class Resnet34Benchmark(TorchImagenetBenchmark):
#
#    name = 'Resnet34Benchmark'
#
#    def run(self):
#        self.run_benchmark('resnet34')
#
#class Resnet50Benchmark(TorchImagenetBenchmark):
#
#    name = 'Resnet50Benchmark'
#
#    def run(self):
#        self.run_benchmark('resnet50')
#
#class VggBenchmark(TorchImagenetBenchmark):
#
#    name = 'VggBenchmark'
#
#    def run(self):
#        self.run_benchmark('vgg')
#
#class Vgg11Benchmark(TorchImagenetBenchmark):
#
#    name = 'Vgg11Benchmark'
#
#    def run(self):
#        self.run_benchmark('vgg11')
#
#class Vgg11BnBenchmark(TorchImagenetBenchmark):
#
#    name = 'Vgg11BnBenchmark'
#
#    def run(self):
#        self.run_benchmark('vgg11_bn')
#
#class Vgg13Benchmark(TorchImagenetBenchmark):
#
#    name = 'Vgg13Benchmark'
#
#    def run(self):
#        self.run_benchmark('vgg13')
#
#class Vgg13BnBenchmark(TorchImagenetBenchmark):
#
#    name = 'Vgg13BnBenchmark'
#
#    def run(self):
#        self.run_benchmark('vgg13_bn')
#
#class Vgg16Benchmark(TorchImagenetBenchmark):
#
#    name = 'Vgg16Benchmark'
#
#    def run(self):
#        self.run_benchmark('vgg16')
#
#class Vgg16BnBenchmark(TorchImagenetBenchmark):
#
#    name = 'Vgg16BnBenchmark'
#
#    def run(self):
#        self.run_benchmark('vgg16_bn')
#
#class Vgg19Benchmark(TorchImagenetBenchmark):
#
#    name = 'Vgg19Benchmark'
#
#    def run(self):
#        self.run_benchmark('vgg19')
#
#class Vgg19BnBenchmark(TorchImagenetBenchmark):
#
#    name = 'Vgg19BnBenchmark'
#
#    def run(self):
#        self.run_benchmark('vgg19_bn')


