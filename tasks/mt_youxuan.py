#from transformers import T5Tokenizer

# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files this script will use the first column for the full texts and the second column for the
# summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
# args.arg_paths.train_file

import json
import os
import datasets
from datasets import DownloadManager, DatasetInfo

_DESCRIPTION = """\
MT dataset covers five business areas: maicai, maoyanyanchu, taxi-yonghu, youxuan, and waimai. The main input content comes from the dialogue between the user and the customer service, the summary comes from the solution, and the information has been desensitized
"""

_FOLDER_PATH = "/home/disk1/grt2021/workspace/UnifiedSKG/data/sample_datas_wo_prefix/single_domain/youxuan/"
_TRAINING_FILE = "train.json"
_VALIDATION_FILE = "valid.json"
_TEST_FILE = "test.json"


class MT_SUMMARY(datasets.GeneratorBasedBuilder):

    def _info(self) -> DatasetInfo:
        """
            info方法，定义数据集的信息，这里要对数据的字段进行定义
        :return:
        """
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                    "text": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                })
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """
            返回datasets.SplitGenerator
            涉及两个参数：name和gen_kwargs
            name: 指定数据集的划分
            gen_kwargs: 指定要读取的文件的路径，与_generate_examples的入参数一致
        :param dl_manager:
        :return: [ datasets.SplitGenerator ]
        """
        _FOLDER_PATH = os.environ['data_folder_path']
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": _FOLDER_PATH + _TRAINING_FILE}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": _FOLDER_PATH + _VALIDATION_FILE}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": _FOLDER_PATH + _TEST_FILE})
        
        ]

    def _generate_examples(self, filepath):
        """
            生成具体的样本，使用yield
            需要额外指定key，id从0开始自增就可以
        :param filepath:
        :return:
        """
        # Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            id_ = 0
            for line in f.readlines():
                example = json.loads(line)
                id_ += 1

                yield id_, {
                    "text": example["text"],
                    "summary": example["summary"]
                }