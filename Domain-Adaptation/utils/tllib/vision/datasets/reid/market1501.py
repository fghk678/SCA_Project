#credits https://github.com/thuml/Transfer-Learning-Library
from .basedataset import BaseImageDataset
from typing import Callable
from PIL import Image
import os
import os.path as osp
import glob
import re
from tllib.vision.datasets._util import download


class Market1501(BaseImageDataset):
    """Market1501 dataset from `Scalable Person Re-identification: A Benchmark (ICCV 2015)
    <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7410490>`_.

    Dataset statistics:
        - identities: 1501 (+1 for background)
        - images: 12936 (train) + 3368 (query) + 15913 (gallery)
        - cameras: 6

    Args:
        root (str): Root directory of dataset
        verbose (bool, optional): If true, print dataset statistics after loading the dataset. Default: True
    """
    dataset_dir = 'Market-1501-v15.09.15'
    archive_name = 'Market-1501-v15.09.15.tgz'
    dataset_url = 'https://cloud.tsinghua.edu.cn/f/29e5f015a7314531b645/?dl=1'

    def __init__(self, root, verbose=True):
        super(Market1501, self).__init__()
        download(root, self.dataset_dir, self.archive_name, self.dataset_url)
        self.relative_dataset_dir = self.dataset_dir
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, cid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= cid <= 6
            cid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, cid))

        return dataset

    def translate(self, transform: Callable, target_root: str):
        """ Translate an image and save it into a specified directory

        Args:
            transform (callable): a transform function that maps images from one domain to another domain
            target_root (str): the root directory to save images

        """
        os.makedirs(target_root, exist_ok=True)
        translated_dataset_dir = osp.join(target_root, self.relative_dataset_dir)

        translated_train_dir = osp.join(translated_dataset_dir, 'bounding_box_train')
        translated_query_dir = osp.join(translated_dataset_dir, 'query')
        translated_gallery_dir = osp.join(translated_dataset_dir, 'bounding_box_test')

        print("Translating dataset with image to image transform...")
        self.translate_dir(transform, self.train_dir, translated_train_dir)
        self.translate_dir(None, self.query_dir, translated_query_dir)
        self.translate_dir(None, self.gallery_dir, translated_gallery_dir)
        print("Translation process is done, save dataset to {}".format(translated_dataset_dir))

    def translate_dir(self, transform, origin_dir: str, target_dir: str):
        image_list = os.listdir(origin_dir)
        for image_name in image_list:
            if not image_name.endswith(".jpg"):
                continue
            image_path = osp.join(origin_dir, image_name)
            image = Image.open(image_path)
            translated_image_path = osp.join(target_dir, image_name)
            translated_image = image
            if transform:
                translated_image = transform(image)

            os.makedirs(os.path.dirname(translated_image_path), exist_ok=True)
            translated_image.save(translated_image_path)
