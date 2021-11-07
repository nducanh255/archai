# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .providers.cifar10_provider import Cifar10Provider
from .providers.cifar100_provider import Cifar100Provider
from .providers.fashion_mnist_provider import FashionMnistProvider
from .providers.imagenet_provider import ImagenetProvider
from .providers.mnist_provider import MnistProvider
from .providers.svhn_provider import SvhnProvider
from .providers.food101_provider import Food101Provider
from .providers.mit67_provider import Mit67Provider
from .providers.sport8_provider import Sport8Provider
from .providers.flower102_provider import Flower102Provider
from .providers.cub200_provider import Cub200Provider
from .providers.cars196_provider import Cars196Provider
from .providers.dogs120_provider import Dogs120Provider
from .providers.nabirds555_provider import NABirds555Provider
from .providers.aircraft_provider import AircraftProvider
from .providers.aircraft_bing_provider import AircraftBingProvider
from .providers.imagenet_downsampled_provider import ImageNet32, ImageNet64
from .providers.simclr_cifar100provider import SimClrCifar100Provider
from .providers.simclr_cifar10provider import SimClrCifar10Provider
from .providers.simclr_flower102provider import SimClrFlower102Provider
from .providers.simclr_cub200provider import SimClrCub200Provider
from .providers.simclr_cars196provider import SimClrCars196Provider
from .providers.simclr_dogs120provider import SimClrDogs120Provider
from .providers.simclr_nabirds555provider import SimClrNABirds555Provider
from .providers.simclr_aircraftprovider import SimClrAircraftProvider
from .providers.simclr_mnistprovider import SimClrMnistProvider
from .providers.simclr_fashion_mnistprovider import SimClrFashionMnistProvider
from .providers.simclr_food101provider import SimClrFood101Provider
from .providers.simclr_mit67provider import SimClrMit67Provider
from .providers.simclr_sport8provider import SimClrSport8Provider
from .providers.simclr_svhnprovider import SimClrSvhnProvider
from .providers.simclr_imagenetprovider import SimClrImageNetProvider
from .providers.simclr_inat21provider import SimClrINat21Provider
from .providers.simclr_inat21miniprovider import SimClrINat21MiniProvider
from .providers.simclr_imagenet_downsampledprovider import SimClrImageNet32Provider, SimClrImageNet64Provider


