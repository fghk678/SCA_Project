#credits https://github.com/thuml/Transfer-Learning-Library
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Caltech101(ImageList):
    """`The Caltech101 Dataset <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ contains objects
    belonging to 101 categories with about 40 to 800 images per category. Most categories have about 50 images.
    The size of each image is roughly 300 x 200 pixels.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d6d4b813a800403f835e/?dl=1"),
        ("train", "train.tgz", "https://cloud.tsinghua.edu.cn/f/ed4d0de80da246f98171/?dl=1"),
        ("test", "test.tgz", "https://cloud.tsinghua.edu.cn/f/db1c444200a848799683/?dl=1")
    ]

    def __init__(self, root, split='train', download=True, **kwargs):
        classes = ['accordion', 'airplanes', 'anchor', 'ant', 'background_google', 'barrel', 'bass', 'beaver',
                   'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon',
                   'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face',
                   'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin',
                   'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'faces', 'faces_easy',
                   'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano',
                   'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree',
                   'kangaroo', 'ketch', 'lamp', 'laptop', 'leopards', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly',
                   'menorah', 'metronome', 'minaret', 'motorbikes', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda',
                   'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner',
                   'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus',
                   'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly',
                   'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Caltech101, self).__init__(root, classes, os.path.join(root, 'image_list', '{}.txt'.format(split)),
                                         **kwargs)
