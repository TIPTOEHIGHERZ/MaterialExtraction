import torch
import random
from typing import Callable, Iterable
import copy
import numpy as np

from utils.io import load_image, save_image, load_exr
import os


class DataLoader:
    image_posfix = ('.jpg', '.png', '.jpeg')
    def __init__(self, transforms: Callable=None):
        self.batch_size = None
        self.device = 'cpu'
        self.files = None
        self.total_images = 0
        self.cnt = 0
        self.transforms = transforms

        return
    
    def shuffle(self):
        random.shuffle(self.files)
        return self
    
    def __getitem__(self, item):
        image = load_image(self.files[item], device=self.device)

        if self.transforms is not None:
            if isinstance(image, list):
                image = [self.transforms(img) for img in image]
            else:
                image = self.transforms(image)

        if isinstance(item, slice):
            return image, self.files[item]
        else:
            return image, self.files[item]

    def get(self, start: int, stop: int):
        assert stop > start, f'stop: {stop}, should be greater than start: {start}'
        return self[start: stop]

    def __iter__(self):
        return self

    def __next__(self):
        assert self.batch_size is not None, 'batch_size is not specified!'
        if self.cnt >= self.total_images:
            raise StopIteration

        images, file_path = self.get(self.cnt, min(self.cnt + self.batch_size, self.total_images))
        self.cnt += self.batch_size

        if len(images) == 0:
            error_files = ','.join(file_path)
            raise FileNotFoundError(f'Fail to load files {error_files}')
        return images, file_path
    
    def __len__(self):
        self.total_images = len(self.files)
        return self.total_images
    
    def total(self):
        return self.total_images


class DTDLoader(DataLoader):
    def __init__(self, fp: str, batch_size=32, relative_dir='dtd/images', device='cpu'):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.relative_dir = relative_dir
        self.fp = os.path.join(fp, relative_dir)
        self.dirs = list()
        for d in os.listdir(self.fp):
            if os.path.isdir(os.path.join(self.fp, d)):
                self.dirs.append(d)

        self.cnt = 0
        self.files_dict = dict()
        self.files = list()
        for d in self.dirs:
            files = [os.path.join(self.fp, d, f) for f in os.listdir(os.path.join(self.fp, d))]
            self.files_dict[d] = list()
            for f in files:
                _, ext = os.path.splitext(f)
                if ext not in self.image_posfix:
                    print(f'{ext} is not supported image type!')
                # if not str(f).endswith(self.image_posfix):
                #     _, ext = os.path.splitext(f)
                #     print(f'{ext} is not supported image type!')
                else:
                    self.files_dict[d].append(f)

            self.files += self.files_dict[d]

        self.total_images = len(self.files)
        return


class KTHLoader(DTDLoader):
    def __init__(self, fp: str, batch_size=32, relative_dir='KTH_TIPS', device='cpu'):
        super().__init__(fp, batch_size, relative_dir, device)
        
        return
    

class KTHaLoader(DataLoader):
    def __init__(self, fp: str, batch_size=32, relative_dir='KTH-TIPS2-a', device='cpu'):
        super().__init__()
        
        self.device = device
        self.batch_size = batch_size
        self.relative_dir = relative_dir
        self.fp = os.path.join(fp, relative_dir)
        self.dirs = list()
        for d in os.listdir(self.fp):
            if os.path.isdir(os.path.join(self.fp, d)):
                self.dirs.append(d)

        self.cnt = 0
        self.files_dict = dict()
        self.files = list()
        for d in self.dirs:
            sub_dirs = os.listdir(os.path.join(self.fp, d))
            for sub_d in sub_dirs:
                abs_path = os.path.join(self.fp, d, sub_d)
                files = [os.path.join(abs_path, f) for f in os.listdir(abs_path)]
                self.files_dict[d] = list()
                for f in files:
                    _, ext = os.path.splitext(f)
                    if ext not in self.image_posfix:
                        print(f'{ext} is not supported image type!')
                    else:
                        self.files_dict[d].append(f)

                self.files += self.files_dict[d]

        self.total_images = len(self.files)

        return


class KTHbLoader(KTHaLoader):
    def __init__(self, fp, batch_size=32, relative_dir='KTH-TIPS2-b', device='cpu'):
        super().__init__(fp, batch_size, relative_dir, device)

        return
    

class PexelLoader(DataLoader):
    def __init__(self, fp: str, batch_size=32, relative_dir='pexels', transforms: Callable=None, device='cpu'):
        super().__init__(transforms=transforms)
        self.device = device
        self.batch_size = batch_size
        self.relative_dir = relative_dir
        self.fp = os.path.join(fp, relative_dir)

        self.cnt = 0
        self.files_dict = dict()
        self.files = list()

        files = os.listdir(self.fp)
        for f in files:
            _, ext = os.path.splitext(f)
            if ext not in self.image_posfix:
                print(f'{ext} is not supported image type!')
            else:
                self.files.append(os.path.join(self.fp, f))
                
        self.total_images = len(self.files)
        return

class ManyTexureLoader(PexelLoader):
    def __init__(self, fp: str, batch_size=32, relative_dir='manytextures', device='cpu'):
        super().__init__(fp, batch_size, relative_dir, device)
        return
    

class AmbientcgLoader(DataLoader):
    name_postfix = ['Color', 'Displacement']

    def __init__(self, fp, batch_size=32, relative_dir='ambientcg_textures/textures', device='cpu'):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.relative_dir = relative_dir
        self.fp = os.path.join(fp, relative_dir)
        self.dirs = list()
        for d in os.listdir(self.fp):
            if os.path.isdir(os.path.join(self.fp, d)):
                self.dirs.append(d)

        self.cnt = 0
        self.files_dict = dict()
        self.files = list()
        for d in self.dirs:
            files = [os.path.join(self.fp, d, f) for f in os.listdir(os.path.join(self.fp, d))]
            self.files_dict[d] = list()
            for f in files:
                name, ext = os.path.splitext(f)
                if ext in self.image_posfix and name.endswith(tuple(self.name_postfix)):
                    self.files_dict[d].append(f)
                # else:
                #     print(f'Discard files {f}')
                
            self.files += self.files_dict[d]

        self.total_images = len(self.files)
        return
    

class TextureDataLoader(DataLoader):
    def __init__(self, data_loaders: list[DataLoader], shuffle=True):
        super().__init__()
        self.files = list()
        self.batch_size = 1

        for dl in data_loaders:
            self.files += dl.files
        
        if shuffle:
            self.shuffle()
            
        self.total_images = len(self.files)
        return

    def __getitem__(self, item):
        if isinstance(item, slice):
            return load_image(self.files[item], device=self.device, to_batch=False)
        else:
            return load_image(self.files[item], device=self.device, to_batch=False)


class LabelDataLoader(DataLoader):
    image_posfix = ('.jpg', '.png', '.jpeg', '.JPEG')
    def __init__(self, fp: str, batch_size=32, relative_dir=None, device='cpu', shuffle=True, transforms: Callable=None):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.relative_dir = relative_dir
        self.fp = fp if relative_dir is None else os.path.join(fp, relative_dir)
        self.files = list()
        self.transforms = transforms

        directory = list()
        for dir_ in os.listdir(self.fp):
            if os.path.isdir(os.path.join(self.fp, dir_)):
                directory.append(os.path.join(self.fp, dir_))

        self.name2label = {dir_: i for i, dir_ in enumerate(directory)}
        self.label2name = {i: os.path.basename(dir_) for i, dir_ in enumerate(directory)}

        for dir_ in directory:
            files = os.listdir(dir_)
            
            for f in files:
                _, ext = os.path.splitext(f)
                if ext not in self.image_posfix:
                    print(f'{ext} is not supported image type!')
                else:
                    self.files.append(os.path.join(dir_, f))
                
        self.total_images = len(self.files)

        if shuffle:
            self.shuffle()

        return
    
    def __getitem__(self, item):
        image = load_image(self.files[item], device=self.device)
        if self.transforms is not None:
            if isinstance(image, list):
                image = [self.transforms(img) for img in image]
            else:
                image = self.transforms(image)

        if isinstance(item, slice):
            return image, torch.tensor([self.name2label[os.path.dirname(f)] for f in self.files[item]], device=self.device)
        else:
            return image, torch.tensor(self.name2label[os.path.dirname(self.files[item])], device=self.device)
        

class ImageDataLoader:
    def __init__(self, image_path: str | list[str], transforms: Callable=None):
        self.image_path = image_path if isinstance(image_path, list) else [image_path]
        self.transforms = transforms

        return
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, item):
        image = load_image(self.image_path[item])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, torch.tensor([item])
    

class RenderLoader:
    image_posfix = ('.jpg', '.png', '.jpeg')
    def __init__(self):
        self.batch_size = None
        self.device = 'cpu'
        self.files = None
        self.cnt = 0

        return

    def shuffle(self):
        random.shuffle(self.files)
        return self
    
    def __getitem__(self, item):
        return self.files[item]

    def get(self, start: int, stop: int):
        assert stop > start, f'stop: {stop}, should be greater than start: {start}'
        return self[start: stop]

    def __iter__(self):
        it = copy.deepcopy(self)
        it.cnt = 0
        return it

    def __next__(self):
        if self.cnt >= self.total_images:
            self.cnt = 0
            raise StopIteration
        
        if len(self.files) == 0:
            raise FileNotFoundError

        file_path = self.files[self.cnt]
        self.cnt += 1
        
        return file_path
    
    def __len__(self):
        # return self.total_images // self.batch_size + int((self.total_images % self.batch_size) != 0)
        return len(self.files)
    
    @property
    def total_images(self):
        return len(self.files)


class PBRTextureDataLoader(RenderLoader):
    image_posfix = ('.jpg', '.png', '.jpeg', '.mtlx', '.usdc')
    image_names = ('image', 'mask')
    # file_type = ('Color', 'NormalGL', 'Metallic', 'Roughness', 'Height')
    file_type = ('Color', 'NormalGL', 'Roughness', 'Height')

    def __init__(
        self, 
        fp: str, 
        gt_fp: str, 
        transforms: dict[Callable]=dict(), 
        fetch_attr=('Color',), 
        transform_group: dict=dict(), 
        selected_files=None,
        good_examples=None,
        gt_has_subdir=False,
        gt_mapping: dict=None,
        max_instance_per_sample=None,
    ):
        super().__init__()
        
        self.fp = fp
        self.gt_fp = gt_fp
        self.transforms = transforms
        self.fetch_attr = fetch_attr
        self.transform_group = transform_group
        self.selected_files = selected_files
        self.good_examples = good_examples

        for key in transform_group.keys():
            assert key in self.transforms.keys()

        gt_mapping_length = set() if gt_mapping is None else {len(key.split('_')) for key in gt_mapping.keys()}
        
        # get ground truth mapping
        if gt_fp is not None:
            self.gt_files = dict()

            if gt_has_subdir:
                target_dirs = list()
                for sub_dir in os.listdir(gt_fp):
                    target_dirs.extend([os.path.join(sub_dir, d) for d in os.listdir(os.path.join(gt_fp, sub_dir))])
            else:
                target_dirs = os.listdir(gt_fp)

            for sub_dirs in target_dirs:
                file_dict = dict()
                file_name = None
                for file in os.listdir(os.path.join(gt_fp, sub_dirs)):
                    if os.path.isdir(file):
                        continue

                    name, ext = os.path.splitext(file)
                    if gt_mapping is None:
                        key = name.split('_')[-1]
                    else:
                        key = None
                        splits = name.split('_')
                        for l in gt_mapping_length:
                            key = None if len(splits) < l else '_'.join(splits[-l:])
                            if key in gt_mapping.keys():
                                name += f'_{gt_mapping[key]}'
                        file_name = os.path.basename(sub_dirs)

                    if name.endswith(self.file_type):
                        file_dict[name.split('_')[-1]] = os.path.join(os.path.join(gt_fp, sub_dirs, file))
                        file_name = '_'.join(name.split('_')[:-1]) if file_name is None else file_name

                assert file_name is not None, f'{sub_dirs} has failed'
                self.gt_files[file_name] = file_dict
        
        type_dirs = [os.path.join(self.fp, d) for d in os.listdir(self.fp) if os.path.isdir(os.path.join(self.fp, d)) \
                     and (selected_files is None or d in selected_files) and (good_examples is None or d in good_examples)]

        self.cnt = 0

        self.files = list()
        self.key_index = dict()
        for type_d in type_dirs:
            sub_dirs = [os.path.join(type_d, d) for d in os.listdir(type_d) if os.path.isdir(os.path.join(type_d, d))]
            for sub_dir in sub_dirs[:max_instance_per_sample] if max_instance_per_sample is not None else sub_dirs:
                files = os.listdir(sub_dir)
                
                file_mapping = {os.path.splitext(file)[0]: file for file in files}
                file_dict = dict()

                for key in file_mapping.keys():         
                    if key in self.image_names:
                        file_dict[key] = os.path.join(sub_dir, file_mapping[key])

                if len(file_dict) != len(self.image_names):
                    for k in self.image_names:
                        if k not in file_dict.keys():
                            print(f'not able to find key {k}!')

                if gt_fp is not None:
                    # could be wrong
                    gt_name = os.path.basename(os.path.splitext(type_d)[0])
                    if gt_name in self.gt_files.keys():
                        # gt_name = '_'.join(gt_name.split('_')[:-1])
                        gt_dict = self.gt_files[gt_name]
                    else: 
                        raise FileNotFoundError(f'not able to find {gt_name}')
                    
                    for key in self.file_type:
                        file_dict[key] = gt_dict[key]
                else:
                    # TODO need to be update
                    pass
                    # assert 'gt' in file_mapping.keys()
                    # file_dict['gt'] = os.path.join(sub_dir, file_mapping['gt'])
                self.files.append(file_dict)
                
                file_key = os.path.basename(type_d)
                if file_key in self.key_index.keys():
                    self.key_index[file_key].append(file_dict)
                else:
                    self.key_index[file_key] = [file_dict]
        
        self.gt_files = list(self.gt_files.values())
        return
    
    def __getitem__(self, item):
        def get_data(file_dict):
            data = {
                key: load_image(file_dict[key]) if file_dict[key].endswith(self.image_posfix) else file_dict[key] for key in self.image_names 
            }

            attrs = {
                attr: load_image(file_dict[attr]) for attr in self.fetch_attr
            }

            data.update(attrs)

            for transform_type in self.transform_group.keys():
                attrs_to_transform = {key: value for key, value in data.items() if key in self.transform_group[transform_type]}
                if len(attrs_to_transform) == 0:
                    continue

                split_range = np.array([value.shape[0] for value in attrs_to_transform.values()])
                split_end = np.cumsum(split_range)
                split_start = split_end - split_range

                value_concatenated = torch.concat([value for value in attrs_to_transform.values()], dim=0)
                # make sure all images have same shape
                value_transformed = self.transforms[transform_type](value_concatenated)
                transformed = {
                    key: value_transformed[split_start[i]: split_end[i]] for i, key in enumerate(attrs_to_transform.keys())
                }
                data.update(transformed)
            
            return data
                    
        if isinstance(item, Iterable) or isinstance(item, slice):
            if isinstance(item, str):
                files = self.key_index[item]
            else:
                files = [self.files[i] for i in item] if isinstance(item, Iterable) else self.files[item]
            data_raw = [get_data(file) for file in files]

            if isinstance(item, str):
                data = data_raw
            else:
                keys = data_raw[0].keys()
                data = dict()
                for key in keys:
                    data[key] = torch.concat([dr[key].unsqueeze(0) for dr in data_raw]) if isinstance(data_raw[0][key] ,torch.Tensor) else \
                        [dr[key] for dr in data_raw]
        else:
            data = get_data(self.files[item])
        
        return data


class RenderResultLoader(RenderLoader):
    image_posfix = ('.jpg', '.png', '.jpeg', '.mtlx', '.usdc')
    image_names = ('image', 'mask', 'image_nolight')
    file_type = ('Color', 'NormalGL', 'Metallic', 'Roughness')
    def __init__(self, fp: str, transform=None):
        super().__init__()

        self.fp = fp
        self.transform = transform

        type_dirs = [os.path.join(self.fp, d) for d in os.listdir(self.fp) if os.path.isdir(os.path.join(self.fp, d))]
        self.cnt = 0

        self.files = list()
        for type_d in type_dirs:
            sub_dirs = [os.path.join(type_d, d) for d in os.listdir(type_d) if os.path.isdir(os.path.join(type_d, d))]
            for sub_dir in sub_dirs:
                files = os.listdir(sub_dir)
                
                file_mapping = {os.path.splitext(file)[0]: file for file in files}
                file_dict = dict()

                for key in file_mapping.keys():                  
                    if key in self.image_names:
                        file_dict[key] = os.path.join(sub_dir, file_mapping[key])

                if len(file_dict) != len(self.image_names):
                    for k in self.image_names:
                        if k not in file_dict.keys():
                            print(f'not able to find key {k}!')

                self.files.append(file_dict)

        return
    
    def __getitem__(self, item):
        def get_data(file_dict):
            image = load_image(file_dict['image'])
            image_nolight = load_image(file_dict['image_nolight'])
            mask = load_image(file_dict['mask'])

            if self.transform is not None:
                # TODO 这里的mask的transform原来是存在问题的，但是可以跑出效果
                # image = self.transform(image)
                # mask = self.transform(mask)
                image_mask_nolight = self.transform(torch.concat([image, mask, image_nolight], dim=0))
                image = image_mask_nolight[:image.shape[0]]
                mask = image_mask_nolight[image.shape[0]:image.shape[0] + mask.shape[0]]
                image_nolight = image_mask_nolight[image.shape[0] + mask.shape[0]: ]

            return image, image_nolight, mask

        if isinstance(item, Iterable):
            files = [self.files[i] for i in item]
            data = [[file.unsqueeze(0) for file in get_data(file_d)] for file_d in files]
            image, image_nolight, mask = list(zip(*data))
        else:
            image, image_nolight, mask = get_data(self.files[item])
        
        return image, image_nolight, mask


class PolyLoader(RenderLoader):
    mapping = {
        'diff': 'Color',
        'disp': 'Height',
        'gl': 'NormalGL',
        'rough': 'Roughness'
    }
    image_posfix = ('.jpg', '.png', '.jpeg')
    supported_posfix = image_posfix + ('.yaml',)
    image_names = ('image', 'mask')
    base_names = ('meta_data',)
    file_type = ('Color', 'NormalGL', 'Roughness', 'Height')

    def __init__(self, fp, transform=None):
        super().__init__()

        self.fp = fp
        self.transform = transform
        self.files = list()

        self.dirs = [d for d in os.listdir(fp) if os.path.isdir(os.path.join(fp, d))]
        for d in self.dirs:
            files = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(fp, d)) if f.endswith(self.supported_posfix)}

            for n in self.image_names:
                assert n in files.keys(), f'{n} not in {files.keys()}'

            file_dict = {
                n: os.path.join(fp, d, files[n]) for n in (self.image_names + self.base_names)
            }

            for key, value in files.items():
                key = key.split('_')
                if len(key) < 2:
                    continue
                key = key[-2]
                if not key in self.mapping.keys():
                    continue
                file_dict[self.mapping[key]] = os.path.join(fp, d, value)
            
            for n in self.file_type:
                assert n in file_dict.keys()
            
            self.files.append(file_dict)

        return
            
    def __getitem__(self, item):
        def get_data(file_dict):
            data = {
                key: load_image(file_dict[key]) if file_dict[key].endswith(self.image_posfix) else file_dict[key] for key in self.image_names 
            }

            attrs = {
                attr: load_image(file_dict[attr]) for attr in self.file_type
            }

            base = {
                key: file_dict[key] for key in self.base_names
            }

            data.update(attrs)
            data.update(base)
            
            return data
                    
        if isinstance(item, Iterable) or isinstance(item, slice):
            if isinstance(item, str):
                files = self.key_index[item]
            else:
                files = [self.files[i] for i in item] if isinstance(item, Iterable) else self.files[item]
            data_raw = [get_data(file) for file in files]

            if isinstance(item, str):
                data = data_raw
            else:
                keys = data_raw[0].keys()
                data = dict()
                for key in keys:
                    data[key] = torch.concat([dr[key].unsqueeze(0) for dr in data_raw]) if isinstance(data_raw[0][key] ,torch.Tensor) else \
                        [dr[key] for dr in data_raw]
        else:
            data = get_data(self.files[item])
        
        return data


class TestPolyLoader(RenderLoader):
    mapping = {
        'diff': 'Color',
        'disp': 'Height',
        'gl': 'NormalGL',
        'rough': 'Roughness'
    }
    support_ext = ('.exr', '.jpg', '.png')

    def __init__(self, fp: str, transform=None):
        super().__init__()

        self.fp = fp
        self.transform = transform

        material_class = [file for file in os.listdir(self.fp) if os.path.isdir(os.path.join(self.fp, file))]
        
        self.files = list()
        for mat in material_class:
            
            instances = [os.path.join(self.fp, mat, file) for file in os.listdir(os.path.join(self.fp, mat)) if os.path.isdir(os.path.join(self.fp, mat, file))]
            for instance in instances:
                file_dict = dict()
                files = [file for file in os.listdir(instance) if os.path.splitext(file)[1] in self.support_ext]

                for file in files:
                    name, ext = os.path.splitext(file)
                    if name == 'reference_clip':
                        file_dict['image'] = os.path.join(instance, file)
                    elif name == 'mask':
                        file_dict['mask'] = os.path.join(instance, file)
                    elif name == 'reference':
                        pass
                    else:
                        name = file.split('_')[-2]
                        if name not in self.mapping.keys():
                            continue
                        file_type = self.mapping[name]
                        file_dict[file_type] = os.path.join(instance, file)

                assert len(self.mapping.keys()) + 2 == len(file_dict), f'{len(file_dict)}, {file_dict}'
                self.files.append(file_dict)

        return
    
    def __getitem__(self, item):
        def get_file(file_dict: dict):
            file_dict = file_dict.copy()
            data = {
                'image': load_image(file_dict['image']),
                'mask': load_image(file_dict['mask'])
            }

            del file_dict['image'], file_dict['mask']

            attrs = {
                key: load_exr(file_dict[key]) if file_dict[key].endswith('.exr') else load_image(file_dict[key]) for key in file_dict.keys()
            }
            data.update(attrs)

            if self.transform is not None:
                # TODO 这里的transform并没有同步上，但是由于输入本来就是方的，random crop不会影响
                data = {
                    key: self.transform(value) for key, value in data.items()
                }
            
            return data


        if isinstance(item, Iterable) or isinstance(item, slice):
            files = [self.files[i] for i in item] if isinstance(item, Iterable) else self.files[item]
            data_raw = [get_file(file) for file in files]
            keys = data_raw[0].keys()

            data = dict()
            for key in keys:
                data[key] = torch.concat([dr[key].unsqueeze(0) for dr in data_raw])
        else:
            data = get_file(self.files[item])

        return data


class FolderLoader(DataLoader):
    image_posfix = ('.jpg', '.png', '.jpeg')

    def __init__(self, fp: str, transform=None):
        super().__init__()

        self.fp = fp
        self.transform = transform

        files = os.listdir(fp)

        self.files = list()
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in self.image_posfix:
                self.files.append(os.path.join(self.fp, file))

        self.total_images = len(self.files)

        return
    
    def __getitem__(self, item):
        image = load_image(self.files[item])

        if self.transform is not None:
            image = self.transform(image)

        return image
    

if __name__ == '__main__':
    data_loader = TextureDataLoader(fp='./datasets/render_result', gt_fp='datasets/ambientcg_textures/textures')

