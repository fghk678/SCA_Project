#credits https://github.com/thuml/Transfer-Learning-Library
from torchvision.datasets.folder import ImageFolder
import os.path as osp
from ._util import download as download_data, check_exits


class Food101(ImageFolder):
    """`Food-101 <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_ is a dataset
    for fine-grained visual recognition with 101,000 images in 101 food categories.

    Args:
        root (str): Root directory of dataset.
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            train/
            test/
    """
    download_list = [
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/1d7bd727cc1e4ce2bef5/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/7e11992d7495417db32b/?dl=1")
    ]

    def __init__(self, root, split='train', transform=None, download=True):
        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))
        super(Food101, self).__init__(osp.join(root, split), transform=transform)
        self.num_classes = 101
