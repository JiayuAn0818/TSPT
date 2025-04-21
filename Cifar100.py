from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np

from .transform import get_transform

cifar_100_root = '/mnt/data4/jyan/data/'

class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)

def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        dataset.uq_idxs = np.arange(len(dataset.uq_idxs))
        return dataset

    else:

        return None

def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]
    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset

def get_cifar_100_datasets(args):
    all_datasets = {}
    args.total_classes = 100
    init_classes = args.init_classes
    args.per_stage_classes = (args.total_classes-init_classes)// args.stage

    train_transform, test_transform = get_transform(image_size=args.image_size, args=args)

    # Init entire training set
    whole_training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True, download=True)
    test_dataset = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False, download=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=range(init_classes))
    test_dataset_labelled = subsample_classes(deepcopy(test_dataset), include_classes=range(init_classes))
    all_datasets['init_train'] = train_dataset_labelled
    all_datasets['init_test'] = test_dataset_labelled

    for i in range(args.stage):
        train_data = subsample_classes(deepcopy(whole_training_set), include_classes=range(init_classes+i*args.per_stage_classes, init_classes+(i+1)*args.per_stage_classes))
        test_data = subsample_classes(deepcopy(test_dataset), include_classes=range(init_classes+i*args.per_stage_classes, init_classes+(i+1)*args.per_stage_classes))
        all_datasets['stage{}_train'.format(i+1)] = train_data
        all_datasets['stage{}_test'.format(i+1)] = test_data
    if 'imbalance' in args.log_name:
        args.per_stage_classes = [8, 12, 6, 14, 10]
        for i in range(args.stage):
            train_data = subsample_classes(deepcopy(whole_training_set), include_classes=range(init_classes+sum(args.per_stage_classes[:i]), init_classes+sum(args.per_stage_classes[:i+1])))
            test_data = subsample_classes(deepcopy(test_dataset), include_classes=range(init_classes+sum(args.per_stage_classes[:i]), init_classes+sum(args.per_stage_classes[:i+1])))
            all_datasets['stage{}_train'.format(i+1)] = train_data
            all_datasets['stage{}_test'.format(i+1)] = test_data

    return all_datasets



if __name__ == '__main__':

    x = get_cifar_100_datasets(None, None, split_train_val=False,
                         train_classes=range(80), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')