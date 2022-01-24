from data.datasets.dukemtmcreid import DukeMTMCreID
from data.datasets.market1501 import Market1501
from data.datasets.cuhk03 import CUHK03
from data.datasets.msmt17 import MSMT17
from data.datasets.veri import VeRi
from data.datasets.dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
