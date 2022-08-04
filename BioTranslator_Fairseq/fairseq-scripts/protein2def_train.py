import pandas as pd
from fairseq_cli import train
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import options
from protein2def_dataset import *
from protein2def_model import *
from protein2def_task import *


def load_fold_data(k, train_fold_file, validation_fold_file):
    train_fold, val_fold = [], []
    for i in range(k):
        train_fold.append(pd.read_pickle(train_fold_file.format(i)))
        val_fold.append(pd.read_pickle(validation_fold_file.format(i)))
    return train_fold, val_fold


def main():
    dataset = 'goa_human_cat'
    training_fold_range = [0]
    data_prefix = '../data/goa_human_terms_10_proteins_fairseq/databin_fold'
    for fold_i in training_fold_range:
        parser = options.get_training_parser()
        src_maxL = 512
        maxL = 512
        parser.set_defaults(arch='protein2def_arch',
                            task='translation',
                            batch_size_valid=16, max_positions=(src_maxL, maxL), fp16='True',
                            memory_efficient_fp16='True',
                            criterion='label_smoothed_cross_entropy_with_alignment',
                            optimizer='adam', lr=[0.0002], batch_size=16,
                            activation_fn='gelu_fast',
                            lr_shrink=0.5,
                            lr_scheduler='reduce_lr_on_plateau',
                            validate_interval=10,
                            update_freq=[1],
                            log_file=data_prefix + '/checkpoints/training_log',
                            save_dir=data_prefix + '/checkpoints')
        args = parser.parse_args()
        cfg = convert_namespace_to_omegaconf(args)
        cfg.task.data = data_prefix
        cfg.task.max_source_positions = src_maxL
        cfg.task.truncate_source=True
        cfg.dataset.skip_invalid_size_inputs_valid_test = True
        train.main(cfg)


if __name__ == '__main__':
    main()