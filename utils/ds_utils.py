import torch
import torchvision
from filelock import FileLock
import os
from natsort import natsorted
import sys

import pickle
import glob
import numpy as np 
import matplotlib.pyplot as plt

import copy 

from torchvision.transforms import v2

from utils.lazy_utils import initialize_vars

from pytorch_lightning import seed_everything

import omegaconf

import random 

import sys
sys.path.append("../")

from utils.args_utils import add_variable_to_hydra_cfg

from torch.utils.data import Sampler

MIN_TB = 80
MAX_TB = 350


def read_all_inside(folder):
    all = natsorted(glob.glob(
        os.path.join(folder, "*")
    ))
    return all 


class Denormalize(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor)
    

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CountAccumulator:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, gt, pred):
        self.tp += int(((gt == 1) & (pred == 1)).sum())
        self.fp += int(((gt == 0) & (pred == 1)).sum())
        self.fn += int(((gt == 1) & (pred == 0)).sum())

    def precision(self): return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")
    def recall(self):    return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")
    def far(self):       return self.fp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")
    def iou(self):       return self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) else float("nan")
    def bias(self):      return (self.tp + self.fp) / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")
    



def pickle_save(file, item):
    import pickle
    import zstandard as zstd

    cctx = zstd.ZstdCompressor(level=10)
    with open(file, 'wb') as fp:
        fp.write(cctx.compress(pickle.dumps(item)))

def pickle_load(file):
    import pickle
    import zstandard as zstd
    from filelock import FileLock

    ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'  # zstd magic bytes

    with FileLock(f'{file}.lck'):
        with open(file, 'rb') as pfile:
            raw = pfile.read()

    if raw[:4] == ZSTD_MAGIC:
        # new format: zstd compressed
        dctx = zstd.ZstdDecompressor()
        data = pickle.loads(dctx.decompress(raw))
    else:
        # legacy format: plain pickle
        data = pickle.loads(raw)

    return data



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None, transform_normalize=None, transform_denormalize=None, cfg=None):
        initialize_vars(self, locals(), exclude=[])

        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform != None:
            x, y = self.transform(x, y)

        return x, y
    
    def __len__(self):
        return len(self.data)

    def get_normalized_data_and_labels(self, device, x=None, y=None):
        if (x != None) and (y != None):
            x, y = self.transform_normalize(x, y) 

        else:
            x, y = self.transform_normalize(self.data, self.targets)

        return x.to(device), y.to(device)
    
    
    def get_unormalized_data_and_labels(self, device, x, y):
        x, y = self.transform_denormalize(x, y)
        return x.to(device), y.to(device)
    

from collections import defaultdict

class MyDatasetFiles(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, transform_normalize=None, transform_denormalize=None, cfg=None):
        self.__dict__.update(locals())
        self.cfg_ds = self.cfg["dataset"]
        self.ds_name = self.cfg_ds["name"]
        print("this is all", root, self.ds_name)
        self.ipc = self.cfg_ds["ipc"]

        print("this is the root", root)
        classes_folder = read_all_inside(root)
        classes_folder = [elt for elt in classes_folder if os.path.isdir(elt)]
        classes_name = [os.path.basename(elt) for elt in classes_folder]
        print("those are the classes", classes_folder)
        self.class_to_idx = self.cfg_ds["class_to_idx"]
        print("this is the class to idx", self.class_to_idx)

    
        self.data = []
        self.targets = []

        print("classes folder and classes name", classes_folder, classes_name)

        for cls_path, cls_name in zip(classes_folder, classes_name):
            if not cls_name in self.class_to_idx:
                print("This dataset should not use", cls_name)
                continue


            print("this is the cls_name", cls_name)
            target = self.class_to_idx[cls_name.lower()]
            data_files = read_all_inside(os.path.join(cls_path, "data"))
            targets = [target] * len(data_files)

            self.data.extend(data_files)
            self.targets.extend(targets)
     
          
        self.samples = list(zip(self.data, self.targets))   # [(path, label), ...]

        self.class_indices = defaultdict(list)               # {label: [global_idx, ...]}
        for i, (_, lbl) in enumerate(self.samples):
            self.class_indices[int(lbl)].append(i)


    def __getitem__(self, idx):
        path, y = self.samples[idx]
        y = torch.tensor(y)
        x = torch.tensor(np.load(path))
        invalid_idx = torch.tensor(np.load(path.replace(
            f"{os.path.sep}data{os.path.sep}",
            f"{os.path.sep}invalid_idx{os.path.sep}")))
        if y == 0:
            x = x[:4]
        x_tpl = (x, invalid_idx)
        if self.transform:
            x, y = self.transform(x_tpl, y)
        y = y.repeat(x.shape[0])     # x still has a variable leading dim
        return x, y

    def __len__(self):
        return len(self.samples)


    def get_unormalized_data_and_labels(self, device, x, y):
        x, y = self.transform_denormalize(x, y)
        return x.to(device), y.to(device)




class BalancedDistributedBatchSampler(Sampler):
    """Each batch = `samples_per_class` indices from every class.
       Reshuffles each epoch; shards whole batches disjointly across ranks."""
    def __init__(self, class_indices, samples_per_class=1,
                 rank=0, world_size=1, seed=0):
        self.class_indices = {k: list(v) for k, v in class_indices.items()}
        self.classes = sorted(self.class_indices.keys())
        self.spc = samples_per_class
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.min_count = min(len(v) for v in self.class_indices.values())
        self.groups_per_epoch = self.min_count // self.spc

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _build_groups(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)        # identical on every rank

        # fresh per-class permutation
        perms = {}
        for c in self.classes:
            idxs = self.class_indices[c]
            order = torch.randperm(len(idxs), generator=g).tolist()
            perms[c] = [idxs[i] for i in order]

        # one balanced group = spc indices from each class
        groups = []
        for gi in range(self.groups_per_epoch):
            batch = []
            for c in self.classes:
                s = gi * self.spc
                batch.extend(perms[c][s:s + self.spc])
            groups.append(batch)

        # shuffle group order too (same generator → same on all ranks)
        order = torch.randperm(len(groups), generator=g).tolist()
        groups = [groups[i] for i in order]

        # drop remainder so every rank gets the same count (no DDP hang)
        usable = (len(groups) // self.world_size) * self.world_size
        return groups[:usable]

    def __iter__(self):
        groups = self._build_groups()
        return iter(groups[self.rank::self.world_size])   # disjoint shard

    def __len__(self):
        usable = (self.groups_per_epoch // self.world_size) * self.world_size
        return usable // self.world_size

def collate_concat(batch):
    xs, ys = zip(*batch)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)




class DataModule():
    def __init__(self, cfg, fabric):
        super().__init__()
        self.cfg = cfg
        self.cfg_ds = cfg["dataset"]
        self.dataset_name = self.cfg_ds["name"]
        self.og_train_shape = None
        self.fabric = fabric
        if self.fabric == None:
            raise ValueError("fabric should not be none")



    def setup(self):
        print("i am inside the setup function")
        cfg = self.cfg
        cfg_ds = cfg["dataset"]

        size = cfg_ds["h"]

        norm_used = self.cfg_ds["norm_used"]
        self.class_to_idx = cfg_ds["class_to_idx"]
        

        if norm_used == "min_max":
            transform_normalize = ComposeWithLabel([
                v2.Lambda(lambda x: (x - MIN_TB) /  (MAX_TB - MIN_TB)),
                v2.Normalize((0.5,), (0.5,)),
            ])
            transform_denormalize = ComposeWithLabel([
                Denormalize((0.5,), (0.5,)),
                v2.Lambda(lambda x: (x  * (MAX_TB - MIN_TB)) + MIN_TB),
                v2.Lambda(lambda x: x.clamp(MIN_TB, MAX_TB)),
            ])

            transform_train = ComposeWithLabel([
                MultipleRandomCrops(s=size, k=self.cfg["k_crops"], cfg=cfg),
                v2.RandomHorizontalFlip(),  
                v2.RandomVerticalFlip(),    
                transform_normalize,
            ])
            transform_test  = ComposeWithLabel([
                transform_normalize,
            ])

        else:
            raise ValueError("Other normalization techniques are no longer implemented")



        # Find dataset root path from possible choices in the list 
        # Faster read time root paths should be first in the list, so we exit once we find a valid path 
        # ------------------------------------------------------------------------
        root_path = cfg["dataset"]["path"]
        # ------------------------------------------------------------------------


        ds_train = MyDatasetFiles(
            root=root_path,
            transform=transform_train, transform_normalize=transform_normalize, transform_denormalize=transform_denormalize,
            cfg=cfg
        )

        sampler_train = BalancedDistributedBatchSampler(
            ds_train.class_indices, samples_per_class=cfg["batch_size"],
            rank=self.fabric.global_rank, world_size=self.fabric.world_size, seed=self.cfg["seed"]
        )

        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_sampler=sampler_train,
            collate_fn=collate_concat,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=False,   # must stay False — see note
            prefetch_factor=10 if cfg["num_workers"] > 0 else None,
        )

        class_dict = cfg_ds["class_to_idx"]

        # Setup quick visual test_tests
        # ---------------------------------------------------------
        test_data_arr = []
        test_targets_arr = []

        sat_c = "f18"
        ds_test = pickle_load(cfg_ds[f"path_test_quick_{sat_c}"])
        for elt in ds_test:
            lon_gpm, lat_gpm, data_gpm = elt["gpm"]["lon"], elt["gpm"]["lat"], elt["gpm"]["data"]
            lon_sat_c, lat_sat_c, data_sat_c = elt[sat_c]["lon"], elt[sat_c]["lat"], elt[sat_c]["data"]

            test_data_arr.append(data_gpm)
            test_targets_arr.append(torch.ones(1) * class_dict["gpm"])
            test_data_arr.append(data_sat_c)
            test_targets_arr.append(torch.ones(1) * class_dict[sat_c])

        # ---------------------------------------------------------

        data_test = test_data_arr
        targets_test = torch.cat(test_targets_arr).to(torch.int64)

        ds_test = MyDataset(
            data=data_test, targets=targets_test, 
            transform=transform_train, transform_normalize=transform_normalize, transform_denormalize=transform_denormalize,
            cfg=cfg
        )
        dl_test = torch.utils.data.DataLoader(
            ds_test, batch_size=100, shuffle=False, drop_last=False, num_workers=cfg["num_workers"], pin_memory=False, persistent_workers=False, prefetch_factor=10,
        )

        return dl_train, sampler_train, dl_test


from abc import ABC, abstractmethod
class LabelTransform(ABC):
    @abstractmethod
    def forward(self, x, y=None):
        pass





class MultipleRandomCrops(torch.nn.Module, LabelTransform):
    def __init__(self, s: int, k: int = 32, cfg=None):
        super().__init__()
        if k < 2 or k % 2 != 0:
            raise ValueError(f"k must be >= 2 and divisible by 2, got k={k}.")
        self.s = s
        self.k = k

        self.cfg = cfg

        self.LABEL_GPM = cfg["dataset"]["class_to_idx"]["gpm"]
        self.LABEL_F18 = cfg["dataset"]["class_to_idx"]["f18"] 



    def forward(self, data_tpl, y=None):
        data, invalid_indices = data_tpl
        s, k = self.s, self.k

        # ── Decide crop size: F18 special-case crops at half resolution ──────
        y_val = y.item() if isinstance(y, torch.Tensor) else y
        cs = s

        # Track input type so we can return the same kind back
        was_tensor = isinstance(data, torch.Tensor)
        if was_tensor:
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data

        C, H, W = data_np.shape
        if cs > H or cs > W:
            raise ValueError(
                f"Crop size cs={cs} exceeds spatial dimensions H={H}, W={W}."
            )

        if not isinstance(invalid_indices, torch.Tensor):
            invalid_indices = torch.as_tensor(invalid_indices, dtype=torch.long)

        half = k // 2

        # ── Prefix sum over invalid rows — O(1) window checks ────────────────
        invalid_mask = torch.zeros(H, dtype=torch.int32)
        if invalid_indices.numel() > 0:
            invalid_mask[invalid_indices] = 1
        prefix = torch.zeros(H + 1, dtype=torch.int32)
        prefix[1:] = invalid_mask.cumsum(0)

        # ── Compute valid anchors once per direction ─────────────────────────
        # increasing: anchor = top row, row_lo = anchor
        anchors_inc = torch.arange(0, H - cs + 1)
        valid_inc = anchors_inc[(prefix[anchors_inc + cs] - prefix[anchors_inc]) == 0]

        # decreasing: anchor = bottom row, row_lo = anchor - cs + 1
        anchors_dec = torch.arange(H - 1, cs - 2, -1)
        valid_dec = anchors_dec[
            (prefix[anchors_dec + 1] - prefix[anchors_dec - cs + 1]) == 0
        ]

        if len(valid_inc) == 0 or len(valid_dec) == 0:
            return None

        # ── Sample half anchors and independent col starts per direction ─────
        sampled_inc = valid_inc[torch.randint(0, len(valid_inc), (half,))]
        sampled_dec = valid_dec[torch.randint(0, len(valid_dec), (half,))]
        col_starts_inc = torch.randint(0, W - cs + 1, (half,))
        col_starts_dec = torch.randint(0, W - cs + 1, (half,))

        # ── Gather crops (at size cs) ────────────────────────────────────────
        crops = np.empty((k, C, cs, cs), dtype=data_np.dtype)
        for i in range(half):
            row_lo = sampled_inc[i].item()
            col_lo = col_starts_inc[i].item()
            crops[i] = data_np[:, row_lo : row_lo + cs, col_lo : col_lo + cs]
        for i in range(half):
            row_lo = sampled_dec[i].item() - cs + 1
            col_lo = col_starts_dec[i].item()
            crops[half + i] = data_np[:, row_lo : row_lo + cs, col_lo : col_lo + cs]

        crops_t = torch.from_numpy(crops)
        crops = crops_t if was_tensor else crops_t.numpy()
        return crops, y

    def extra_repr(self) -> str:
        return f"s={self.s}, k={self.k}"
        


class ComposeWithLabel(LabelTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def forward(self, x, y=None):
        for t in self.transforms:
            if isinstance(t, LabelTransform):
                x, y = t(x, y)
            else:
                x = t(x)
        return x, y

    def __call__(self, x, y=None):
        return self.forward(x, y)

    




