from torch.utils.data import Subset
from torchvision import transforms
from utils.data import build_dataset, pil_loader, normalize_01_into_pm1
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
import os.path as osp

def get_datasets(imagenet_path, num_per_class, classes, reps_per_image, final_reso=256, no_crop=False):
    # from utils.data build_dataset
    mid_reso=1.125
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    resize = transforms.Compose(
        (
            [
                transforms.Resize(mid_reso, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.RandomCrop((final_reso, final_reso))
            ] if not no_crop else [
                transforms.Resize(final_reso, interpolation=transforms.InterpolationMode.LANCZOS),
            ]
        ) + [
            transforms.ToTensor(), normalize_01_into_pm1,
        ]
    )

    train_set = DatasetFolder(root=osp.join(imagenet_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=resize)
    val_set = DatasetFolder(root=osp.join(imagenet_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=resize)

    return (
        repeat_and_subset_per_class(train_set, num_per_class, classes, reps_per_image),
        repeat_and_subset_per_class(val_set, num_per_class, classes, reps_per_image),
    )

def repeat_and_subset_per_class(dataset, num_per_class, classes, repeats):
    from collections import defaultdict
    labels = getattr(dataset, 'targets', None)
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        if label in classes:
            class_to_indices[label].append(idx)
    indices = []
    for label, idxs in class_to_indices.items():
        selected = sorted(idxs)[:num_per_class]
        for idx in selected:
            indices.extend([idx] * repeats) 
    return Subset(dataset, indices)

def install_hooks(model, hook_filter=None, hook_fn=None):
    return HookManager(model, hook_filter, hook_fn)

class HookManager:
    def __init__(self, model, hook_filter=None, hook_fn=None):
        self.model = model
        self.hook_filter = hook_filter
        self.hook_fn = hook_fn
        self.handles = []

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if self.hook_filter and self.hook_filter(module_name, module):
                handle = module.register_forward_hook(lambda m, i, o, name=module_name: self.hook_fn(m, i, o, name))
                self.handles.append(handle)
        return self.handles

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles = []
