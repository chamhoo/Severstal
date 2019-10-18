from math import sqrt
import tensorflow as tf
from fiat.components.arch.res import ResClassifier


def resnet(x, numlayer='18', nstage=5, nclass=1, channels=64, dropout_rate=0.25):
    from fiat.components.arch.res import ResNet
    comp = ResClassifier(
        x=x,
        net=ResNet,
        numlayer=numlayer,
        nstage=nstage,
        nclass=nclass,
        channels=channels,
        dropout_rate=dropout_rate)
    comp.reslayer1()
    comp.resmainlayer()
    return comp.resoutput()


def resnext(**kwargs):
    from fiat.components.arch.res import ResNext
    comp = ResClassifier(
        x=kwargs['x'],
        net=ResNext,
        numlayer=kwargs['numlayer'],
        nstage=kwargs['nstage'],
        nclass=kwargs['nclass'],
        channels=kwargs['channels'],
        dropout_rate=kwargs['dropout_rate'])
    comp.reslayer1()
    comp.resmainlayer()
    return comp.resoutput()


def seresnet(x, numlayer='18', nstage=5, nclass=1, channels=64, dropout_rate=0.25):
    from fiat.components.arch.res import SeResNet
    comp = ResClassifier(
        x=x,
        net=SeResNet,
        numlayer=numlayer,
        nstage=nstage,
        nclass=nclass,
        channels=channels,
        dropout_rate=dropout_rate)
    comp.reslayer1()
    comp.resmainlayer()
    return comp.resoutput()


def seresnext(**kwargs):
    from fiat.components.arch.res import SeResNext
    comp = ResClassifier(
        x=kwargs['x'],
        net=SeResNext,
        numlayer=kwargs['numlayer'],
        nstage=kwargs['nstage'],
        nclass=kwargs['nclass'],
        channels=kwargs['channels'],
        dropout_rate=kwargs['dropout_rate'])
    comp.reslayer1()
    comp.resmainlayer()
    return comp.resoutput()