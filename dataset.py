import os
import imgaug.augmenters as iaa
import cv2
import numpy as np
from torch.utils.data import Dataset
from math import ceil
import numpy.random as npr
from skimage import io
import random
from tqdm import tqdm
import torch
import spatial_aug
from threading import Lock

def make_dataset():
    labels = {}
    with open('./labels.csv', 'r') as fr:
        for ln in fr:
            num, label = ln.split('\t')
            num = int(num)
            label = label.strip()
            try:
                labels[label].append(num)
            except:
                labels[label] = [num]
    print('Found %d labels' % len(labels))
    train_ds = Dataset(labels, 0.7)
    test_ds = Dataset(labels, -0.3)
    return train_ds, test_ds

class Dataset(object):
    def __init__(self, labels, part):
        self.resizer = iaa.Resize({'width' : 112, 'height' : 112})
        self.big_resizer = iaa.Resize({'width' : 224, 'height' : 224})
        self.mean = torch.tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1, 1)
        self.big_mean = torch.tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1, 1)
        self.big_std = torch.tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1, 1)
        self.samples = []
        self.targets = []
        self.batch_size = 64
        self.spatial_aug = spatial_aug.Compose([
            spatial_aug.MultiScaleRandomCrop(),
            spatial_aug.SpatialElasticDisplacement()
        ])
        my_labels = {}
        for label in labels:
            nums = labels[label]
            if part > 0:
                my_labels[label] = nums[:int(len(nums)*part)]
            else:
                my_labels[label] = nums[int(len(nums)*part):]

        for label in my_labels:
            nums = my_labels[label]
            print('Loading %d paths for label %s' % (len(nums), label))
            for _, num in tqdm(enumerate(nums)):
                fls = sorted(os.listdir('./data/%d/' % num))
                if len(fls) < 32:
                    continue
                for ptr in range(32, len(fls) + 1):
                    fls_slice = tuple(map(lambda x : './data/%d/%s' % (num, x), fls[ptr - 32:ptr]))
                    self.samples.append(fls_slice)
                    if label == 'No gesture':
                        self.targets.append(0)
                    else:
                        self.targets.append(1)
        self.order = np.arange(len(self.samples))

    def __getitem__(self, i):
        batch_samples = []
        big_batch_samples = []
        batch_target = []
        if i*self.batch_size >= len(self.samples):
            raise IndexError()
        self.spatial_aug.randomize_parameters()
        for j in range(i*self.batch_size, min((i+1)*self.batch_size, len(self.samples))):
            f_names = self.samples[self.order[j]]
            cv2_imgs = []
            cv2_big_images = []
            for f_name in f_names:
                img = cv2.imread(f_name)
                img = img.reshape((1, img.shape[0], img.shape[1], 1))
                big_img = self.big_resizer(images=img)[0, :, :, :].reshape(1, 224, 224)
                img = self.resizer(images=img)[0, :, :, :].reshape(1, 112, 112)
                cv2_imgs.append(img)
                cv2_big_imgs.append(big_img)
            sample = np.stack(cv2_imgs, axis=1)
            imgs = [self.spatial_aug(sample[:, i, :, :].reshape((112, 112, 1))).reshape((1, 112, 112)) for i in range(32)]
            sample = np.stack(imgs, axis=1)
            big_sample = np.stack(cv2_big_imgs, axis=1)
            big_imgs = [self.spatial_aug(big_sample[:, i, :, :].reshape((224, 224, 1))).reshape((1, 224, 224)) for i in range(32)]
            big_sample = np.stack(imgs, axis=1)
            target = self.targets[self.order[j]]
            batch_samples.append(sample)
            big_batch_samples.append(big_sample)
            batch_target.append(target)
        batch = np.stack(batch_samples)
        big_batch = np.stach(big_batch_samples)
        target = np.array(batch_target)
        batch = torch.from_numpy(batch).float()
        big_batch = torch.from_numpy(big_batch).float()
        target = torch.from_numpy(target).long()
        return (batch/255 - self.mean)/self.std, (big_batch/255 - self.big_mean)/self.big_std, target

    def __len__(self):
        return ceil(len(self.samples)/self.batch_size)

    def shuffle(self):
        npr.shuffle(self.order)

def make_pretrain_datasets(path='/tmp/imagenet/compressed_dataset', train_test=0.8, aug=None, batch_size=64):
    all_parts = filter(lambda x : x.split('.')[0][-6:] == 'images', os.listdir(path))
    all_parts = list(map(lambda x : x.split('.')[0][:-7], all_parts))
    random.shuffle(all_parts)
    all_parts = list(map(lambda x : os.path.join(path, x), all_parts))
    parts_length = len(all_parts)
    train_parts = all_parts[:int(parts_length*train_test)]
    test_parts = all_parts[int(parts_length*train_test):]
    train_dataset = PretrainDataset(train_parts, aug=aug, name='Train dataset', batch_size=batch_size)
    test_dataset = PretrainDataset(test_parts, aug=aug, name='Test dataset', batch_size=batch_size)
    return train_dataset, test_dataset

class PretrainDataset(Dataset):
    def __init__(self, parts, aug=None, resize=(224, 224), name='Dataset', batch_size=64):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1, 1)
        self.batch_size = batch_size
        self.aug = aug
        self.name = name
        self.parts = [[0, part] for part in parts]
        self.length = 0
        for i, (_, part) in enumerate(self.parts):
            new_part = ImagenetPart(600, 600)
            new_part.load(part)
            self.parts[i][0] = len(new_part)
            self.length += len(new_part)
        self.milestones = np.cumsum(list(map(lambda x : x[0], self.parts)), dtype=np.int32)
        self.resizer = iaa.Resize({'width' : resize[0], 'height' : resize[1]})
        self.resizer_size = resize
        self.unpacked_index = None
        print('%s loaded (%d)' % (name, len(self)))

    def _get_frames_from_img(self, img, seq=[]):
        _seq = []
        if seq:
            width = seq[0][0]
            height = seq[0][1]
            begin_x = seq[0][2]
            begin_y = seq[0][3]
        else:
            width = random.randint(75, 150)
            height = random.randint(75, 150)
            begin_x = random.randint(0, self.resizer_size[0] - width)
            begin_y = random.randint(0, self.resizer_size[1] - height)
            _seq.append([width, height, begin_x, begin_y])
        frame = img[:, begin_y:begin_y+height, begin_x:begin_x+width, :]
        frames = [self.resizer(images=frame)[0, :, :, :]]
        for i in range(31):
            if seq:
                width = seq[i + 1][0]
                height = seq[i + 1][1]
                begin_x = seq[i + 1][2]
                begin_y = seq[i + 1][3]
            else:
                shift_x = random.randint(-20, 20)
                shift_y = random.randint(-20, 20)
                width_scale = random.uniform(0.8, 1.2)
                height_scale = random.uniform(0.8, 1.2)
                width *= width_scale
                height *= height_scale
                width, height = list(map(int, [width, height]))
                if width < 75:
                    width = 75
                if height < 75:
                    height = 75
                if width > self.resizer_size[0]:
                    width = self.resizer_size[0]
                if height > self.resizer_size[1]:
                    height = self.resizer_size[1]
                begin_x += shift_x
                begin_y += shift_y
                if begin_x < 0:
                    begin_x = 0
                if begin_y < 0:
                    begin_y = 0
                begin_x -= max(0, begin_x + width - self.resizer_size[0])
                begin_y -= max(0, begin_y + height - self.resizer_size[1])
                _seq.append([width, height, begin_x, begin_y])
            frame = img[:, begin_y:begin_y+height, begin_x:begin_x+width, :]
            frames.append(self.resizer(images=frame)[0, :, :, :])
        frames = np.stack(frames)
        return frames, (_seq or seq)

    def _get_sample(self, part_index, index):
        if self.unpacked_index != part_index:
            self.unpacked_index = part_index
            self.unpacked_part = ImagenetPart(600, 600)
            self.unpacked_part.load(self.parts[part_index][1])
        sample_img = self.unpacked_part[index]
        h, w = sample_img.shape[:2]
        sample_img = sample_img.reshape(1, h, w, 3)
        sample_img = self.resizer(images=sample_img)
        sample_frames, seq = self._get_frames_from_img(sample_img)
        sample = torch.from_numpy(sample_frames).float()
        sample = torch.transpose(sample, 1, 3)
        sample = torch.transpose(sample, 0, 1)
        if self.aug:
            sample_img_aug = self.aug(images=sample_img)
            aug_sample_frames, _ = self._get_frames_from_img(sample_img_aug, seq)
            sample_aug = torch.from_numpy(aug_sample_frames).float()
            sample_aug = torch.transpose(sample_aug, 1, 3)
            sample_aug = torch.transpose(sample_aug, 0, 1)
        else:
            sample_aug = sample
        return sample_aug[:, :, :, :]/255, sample[:, :, :, :]/255

    def __getitem__(self, i):
        aug_samples = []
        samples = []
        for j in range(i*self.batch_size, min((i+1)*self.batch_size, self.length)):
            part_index = 0
            while i >= self.milestones[part_index]:
                part_index += 1
            index = i
            if part_index > 0:
                index -= self.milestones[part_index - 1]
            aug_sample, sample = self._get_sample(part_index, index)
            aug_samples.append(aug_sample)
            samples.append(sample)
        if len(aug_samples) == 0:
            raise IndexError()
        aug_batch = torch.stack(aug_samples)
        sample_batch = torch.stack(samples)
        return (aug_batch - self.mean)/self.std, (sample_batch - self.mean)/self.std


    def __len__(self):
        return ceil(self.length/self.batch_size)

    def shuffle(self):
        self.unpacked_index = None
        npr.shuffle(self.parts)
        self.milestones = np.cumsum(list(map(lambda x : x[0], self.parts)), dtype=np.int32)

class ImagenetPart(object):
    def __init__(self, width=1020, height=1020):
        self.images = np.zeros((0, 3, width, height), dtype=np.uint8)
        self.sizes = np.zeros((0, 2), dtype=int)
        self.width = width
        self.height = height
        self.by_height_resizer = iaa.Resize({'width' : 'keep-aspect-ratio', 'height' : self.height})
        self.by_width_resizer = iaa.Resize({'width' : self.width, 'height' : 'keep-aspect-ratio'})
        self.mutex = Lock()

    def add(self, image):
        height, width = image.shape[:2]
        while height > self.height or width > self.width:
            image = image.reshape((1, height, width, 3))
            if height > self.height:
                image = self.by_height_resizer(images=image)
            elif width > self.width:
                image = self.by_width_resizer(images=image)
            image = image[0, :, :, :]
            height, width = image.shape[:2]
        image = np.pad(image, ((0, self.height - height), (0, self.width - width), (0, 0)), 'constant', constant_values=0)
        image = image.reshape((1, self.height, self.width, 3))
        image = np.transpose(image, (0, 3, 2, 1))
        size = np.array((width, height), dtype=int).reshape(1, 2)
        with self.mutex:
            self.images = np.concatenate((self.images, image), axis=0)
            self.sizes = np.concatenate((self.sizes, size))

    def save(self, path='imagenet_dataset'):
        with self.mutex:
            np.save(path + "_images", self.images)
            np.save(path + "_sizes", self.sizes)

    def __len__(self):
        with self.mutex:
            return self.sizes.shape[0]

    def __getitem__(self, i):
        with self.mutex:
            width, height = self.sizes[i, :]
            image = self.images[i, :, :width, :height]
            return np.transpose(image, (2, 1, 0))

    def load(self, path='imagenet_dataset'):
        with self.mutex:
            self.images = np.load(path + "_images.npy")
            self.sizes = np.load(path + "_sizes.npy")
