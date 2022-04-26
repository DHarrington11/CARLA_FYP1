import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader


def load_data(path):
    """
    Helper method to load the dataset
    :param path: Path to location of dataset
    :return: lists of all the images and masks
    """
    images_list = list(path.glob('RGB_test/*.png'))
    images_list.sort()
    masks_list = list(path.glob('semanticArr_test/*.npy'))
    masks_list.sort()
    if len(images_list) != len(masks_list):
        raise ValueError('Invalid data loaded')
    images_list = np.array(images_list)
    masks_list = np.array(masks_list)
    return images_list, masks_list


class SegmentationAgent:
    def __init__(self, val_percentage, test_num, num_classes,
                 batch_size, img_size, data_path, shuffle_data,
                 learning_rate, device):
        """
        A helper class to facilitate the training of the model
        """
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.images_list, self.masks_list = load_data(data_path)
        train_split, val_split, test_split = self.make_splits(
                val_percentage, test_num, shuffle_data)
        #print(train_split)
        self.train_loader = self.get_dataloader(train_split)
        self.validation_loader = self.get_dataloader(val_split)
        self.test_loader = self.get_dataloader(test_split)
        self.model = SegmentationUNet(self.num_classes, self.device)
        self.criterion = TverskyCrossEntropyDiceWeightedLoss(self.num_classes,
                                                             self.device)
        #self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def make_splits(self, val_percentage=0.2, test_num=10, shuffle=True):
        """
        Split the data into train, validation and test datasets
        :param val_percentage: A decimal number which tells the percentage of
                data to use for validation
        :param test_num: The number of images to use for testing
        :param shuffle: Shuffle the data before making splits
        :return: tuples of splits
        """
        if shuffle:
            shuffle_idx = np.random.permutation(range(len(self.images_list)))
            self.images_list = self.images_list[shuffle_idx]
            #print(shuffle_idx)
            self.masks_list = self.masks_list[shuffle_idx]

        val_num = len(self.images_list) - int(
            val_percentage * len(self.images_list))
        
        #print(val_num)

        train_images = self.images_list[:val_num]
        train_masks = self.masks_list[:val_num]

        #print(len(train_images))
        #print(len(train_masks))

        validation_images = self.images_list[val_num:-test_num]
        validation_masks = self.masks_list[val_num:-test_num]

        #print(len(validation_images))
        #print(len(validation_masks))

        test_images = self.images_list[-test_num:]
        test_masks = self.masks_list[-test_num:]

        #print(len(test_images))
        #print(len(test_masks))

        return (train_images, train_masks), \
               (validation_images, validation_masks), \
               (test_images, test_masks)

    def get_dataloader(self, split):
        """
        Create a DataLoader for the given split
        :param split: train split, validation split or test split of the data
        :return: DataLoader
        """
        return DataLoader(SegmentationDataset(split[0], split[1], self.img_size,
                                              self.num_classes, self.device),
                          self.batch_size, shuffle=True)

from numpy import array, moveaxis
from PIL import Image
from torch import from_numpy
from torch.utils.data import Dataset
import imageio as io

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, size, num_classes, device):
        """
        Class extending the PyTorch Dataset class
        :param image_paths: list with paths to images
        :param mask_paths: list with paths to masks
        :param size: size to which the image is resized
        :param num_classes: number of classes to classify
        :param device: device on which the model is trained
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size
        self.num_classes = num_classes
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = array(
                Image.open(self.image_paths[idx]).resize((self.size, self.size),
                                                         resample=Image.BILINEAR))
        image = image / 255
        image = image[:,:,:3]
        #image = image.reshape((128,128,3))
        # mask2 = array(
        #         Image.open(self.mask_paths[idx]).resize((self.size, self.size),
        #                                                 resample=Image.NEAREST),
        #         dtype='int')[:, :, 0]
        mask = np.load(self.mask_paths[idx]) 
        #print(mask)       
        # mask = io.imread(self.mask_paths[idx])
        # mask = mask[:,:,:3]

        # #mask = array(Image.open(self.mask_paths[idx]),dtype='int')[:,:,3]
    
        # for i in range(128):
        #     for j in range(128):
        #         if (mask[i][j]== [0,0,0]).all(): #Unlabeled
        #             mask[i][j][0]=0
        #         elif (mask[i][j]== [70,70,70]).all(): #Building
        #             mask[i][j][0]=1
        #         elif (mask[i][j]== [100,40,40]).all(): #Fence
        #             mask[i][j][0]=2
        #         elif (mask[i][j]== [55,90,80]).all(): #Other
        #             mask[i][j][0]=3
        #         elif (mask[i][j]== [220,20,60]).all(): #Pedestrian
        #             mask[i][j][0]=4
        #         elif (mask[i][j]== [153,153,153]).all(): #Pole
        #             mask[i][j][0]=5
        #         elif (mask[i][j]== [157,234,50]).all(): #Roadline
        #             mask[i][j][0]=6
        #         elif (mask[i][j]== [128,64,128]).all(): #Road
        #             mask[i][j][0]=7
        #         elif (mask[i][j]== [244,35,232]).all(): #Sidewalk
        #             mask[i][j][0]=8
        #         elif (mask[i][j]== [107,142,35]).all(): #vegetation
        #             mask[i][j][0]=9
        #         elif (mask[i][j]== [0,0,142]).all(): #vehicles
        #             mask[i][j][0]=10
        #         elif (mask[i][j]== [102,102,156]).all(): #wall
        #             mask[i][j][0]=11
        #         elif (mask[i][j]== [220,220,0]).all(): #TrafficSign
        #             mask[i][j][0]=12
                    
        #         elif (mask[i][j]== [70, 130, 180]).all(): #sky-> other
        #             mask[i][j][0]=3
        #         elif (mask[i][j]== [81, 0, 81]).all(): #Ground-> Other
        #             mask[i][j][0]=3
        #         elif (mask[i][j]== [150, 100, 100]).all(): #Bridge -> Other
        #             mask[i][j][0]=3
        #         elif (mask[i][j]== [230, 150, 140]).all(): #Rail Track-> Other
        #             mask[i][j][0]=3
        #         elif (mask[i][j]== [180, 165, 180]).all(): #GuardRail-> Other
        #             mask[i][j][0]=3
        #         elif (mask[i][j]== [250, 170, 30]).all(): #Traffic Light-> Other
        #             mask[i][j][0]=3
        #         elif (mask[i][j]== [110, 190, 160]).all(): #Static-> Other
        #             mask[i][j][0]=3
        #         elif (mask[i][j]== [170, 120, 50]).all(): #Dynamic->Other
        #             mask[i][j][0]=3
        #         elif (mask[i][j]== [45,60,150]).all(): #water -> none
        #             mask[i][j][0]=0
        #         elif (mask[i][j]== [145, 170, 100]).all(): #terrain-> Other
        #             mask[i][j][0]=3
        #         else:
        #             print("NOT FOUND: {}".format(mask[i][j]))
        
        # mask = mask[:,:,0]
        
        #print(mask)
        #mask = array(mask.resize((self.size,self.size), resample=Image.NEAREST),dtype='int')[:,:,0]
        
        image = moveaxis(image, -1, 0)
        image = from_numpy(image).float().to(self.device)
        mask = moveaxis(mask, -1, 0)
        mask = from_numpy(mask).long().to(self.device)
        # mask2 = moveaxis(mask2, -1, 0)
        # mask2 = from_numpy(mask2).long().to(self.device)

        # print(mask.shape)
        # print(mask2.shape)
        #print(mask[:,:,0].shape)
        
        # print('image: {}'.format(image))
        # print('mask: {}'.format(mask))

        return image, mask

from torch import cat
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, MaxPool2d, Module, \
    ModuleList, Sequential
from torch.nn.functional import relu


def conv(in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
    """
    A convolution block with a conv layer and batch norm
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of the kernel
    :param padding: number of pixels to pad on all sides
    :param batch_norm: to use batch norm or not
    :return: PyTorch Tensor
    """
    c = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
               padding=padding)
    if batch_norm:
        bn = BatchNorm2d(out_channels)
        return Sequential(c, bn)
    return c


class DownConv(Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        """
        A PyTorch Module to create the downward block of UNet architecture
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param pooling: to use pooling or not
        """
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv_in = conv(self.in_channels, self.out_channels)
        self.conv_out = conv(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = relu(self.conv_in(x))
        x = relu(self.conv_out(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(Module):
    def __init__(self, in_channels, out_channels):
        """
        A PyTorch Module to create the upward block of UNet architecture
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        """
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = ConvTranspose2d(self.in_channels, self.out_channels,
                                      kernel_size=2, stride=2)

        self.conv_in = conv(2 * self.out_channels, self.out_channels)
        self.conv_out = conv(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up, output_size=from_down.size())
        x = cat((from_up, from_down), 1)
        x = relu(self.conv_in(x))
        x = relu(self.conv_out(x))
        return x


class SegmentationUNet(Module):
    def __init__(self, num_classes, device, in_channels=3, depth=5,
                 start_filts=64):
        """
        The UNet model
        :param num_classes: number of classes to segment
        :param device: device on which the model is to be trained
        :param in_channels: number of input channels
        :param depth: the depth of the model
        :param start_filts: number of filters in the starting block
        """
        super(SegmentationUNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.device = device

        self.down_convs = []
        self.up_convs = []

        outs = 0
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.conv_final = conv(outs, self.num_classes, kernel_size=1, padding=0,
                               batch_norm=False)

        self.down_convs = ModuleList(self.down_convs)
        self.up_convs = ModuleList(self.up_convs)

    def forward(self, x):
        x = x.to(self.device)
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        return x

import numpy as np
import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy, softmax


class TverskyCrossEntropyDiceWeightedLoss(Module):
    def __init__(self, num_classes, device):
        """
        A wrapper Module for a custom loss function
        """
        super(TverskyCrossEntropyDiceWeightedLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def tversky_loss(self, pred, target, alpha=0.5, beta=0.5):
        """
        Calculate the Tversky loss for the input batches
        :param pred: predicted batch from model
        :param target: target batch from input
        :param alpha: multiplier for false positives
        :param beta: multiplier for false negatives
        :return: Tversky loss
        """
        target_oh = torch.eye(self.num_classes)[target.squeeze(1)]
        target_oh = target_oh.permute(0, 3, 1, 2).float()
        probs = softmax(pred, dim=1)
        target_oh = target_oh.type(pred.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        inter = torch.sum(probs * target_oh, dims)
        fps = torch.sum(probs * (1 - target_oh), dims)
        fns = torch.sum((1 - probs) * target_oh, dims)
        t = (inter / (inter + (alpha * fps) + (beta * fns))).mean()
        return 1 - t

    def class_dice(self, pred, target, epsilon=1e-6):
        """
        Calculate DICE coefficent for each class
        :param pred: predicted batch from model
        :param target: target batch from input
        :param epsilon: very small number to prevent divide by 0 errors
        :return: list of DICE loss for each class
        """
        pred_class = torch.argmax(pred, dim=1)
        dice = np.ones(self.num_classes)
        for c in range(self.num_classes):
            p = (pred_class == c)
            t = (target == c)
            inter = (p * t).sum().float()
            union = p.sum() + t.sum() + epsilon
            d = 2 * inter / union
            dice[c] = 1 - d
        return torch.from_numpy(dice).float()

    def forward(self, pred, target, cross_entropy_weight=0.5,
                tversky_weight=0.5):
        """
        Calculate the custom loss
        :param pred: predicted batch from model
        :param target: target batch from input
        :param cross_entropy_weight: multiplier for cross entropy loss
        :param tversky_weight: multiplier for tversky loss
        :return: loss value for batch
        """
        if cross_entropy_weight + tversky_weight != 1:
            raise ValueError('Cross Entropy weight and Tversky weight should '
                             'sum to 1')
        ce = cross_entropy(pred, target,
                           weight=self.class_dice(pred, target).to(self.device))
        tv = self.tversky_loss(pred, target)
        loss = (cross_entropy_weight * ce) + (tversky_weight * tv)
        return loss


if __name__ == '__main__':
  SegmentationAgent()