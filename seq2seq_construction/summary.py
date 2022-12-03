import os
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm

class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
Data item is formatted as:
    {
        
        "text": datasets.Value("string"),
        "summary": datasets.Value("string"),
            
        # Note
        # 
        # A datasets.Sequence with a internal dictionary feature will be automatically converted into a dictionary of 
        # lists. This behavior is implemented to have a compatilbity layer with the TensorFlow Datasets library but may be 
        # un-wanted in some cases. If you don’t want this behavior, you can use a python list instead of the datasets.Sequen
        # ce.
        ),
    }
"""


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets
        # cache_path = os.path.join(cache_root, 'summary_train.cache')
        chache_name = args.model.description.strip() 
        cache_path = os.path.join(cache_root, chache_name + '_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.raw_examples, self.full_src_lst, self.full_tgt_lst = torch.load(cache_path)
        else:
            self.raw_examples = []
            self.full_src_lst = []
            self.full_tgt_lst = []
            for example in tqdm(self.raw_datasets):
                
                self.raw_examples.append(example)
                self.full_src_lst.append(example['text'])
                self.full_tgt_lst.append(example['summary'])
            if args.dataset.use_cache:
                torch.save((self.raw_examples, self.full_src_lst, self.full_tgt_lst), cache_path)

    def __getitem__(self, index) -> T_co:
        raw_data = self.raw_examples[index]
        raw_data.update({"struct_in": "",
                         "text_in": self.full_src_lst[index],
                         "seq_out": self.full_tgt_lst[index]})
        return raw_data

    def __len__(self):
        return len(self.full_src_lst)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        # cache_path = os.path.join(cache_root, 'summary_validation.cache')
        chache_name = args.model.description.strip() 
        cache_path = os.path.join(cache_root, chache_name + '_validation.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.full_src_lst, self.full_tgt_lst = torch.load(cache_path)
        else:

            self.full_src_lst = []
            self.full_tgt_lst = []

            for example in tqdm(self.raw_datasets):

                self.full_src_lst.append(example['text'])
                self.full_tgt_lst.append(example['summary'])
                
            if args.dataset.use_cache:
                torch.save((self.full_src_lst, self.full_tgt_lst), cache_path)

    def __getitem__(self, index):
        raw_data = self.raw_datasets[index]
        raw_data.update({"struct_in": "",
                         "text_in": self.full_src_lst[index],
                         "seq_out": self.full_tgt_lst[index]
                         })
        return raw_data

    def __len__(self):
        return len(self.full_src_lst)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        # cache_path = os.path.join(cache_root, 'summary_test.cache')
        chache_name = args.model.description.strip() 
        cache_path = os.path.join(cache_root, chache_name + '_test.cache')        
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.full_src_lst, self.full_tgt_lst = torch.load(cache_path)
        else:

            self.full_src_lst = []
            self.full_tgt_lst = []

            for example in tqdm(self.raw_datasets):

                self.full_src_lst.append(example['text'])
                self.full_tgt_lst.append(example['summary'])
                
            if args.dataset.use_cache:
                torch.save((self.full_src_lst, self.full_tgt_lst), cache_path)

    def __getitem__(self, index):
        raw_data = self.raw_datasets[index]
        raw_data.update({"struct_in": "",
                         "text_in": self.full_src_lst[index],
                         "seq_out": self.full_tgt_lst[index]
                         })
        return raw_data

    def __len__(self):
        return len(self.full_src_lst)