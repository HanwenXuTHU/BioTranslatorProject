import pandas as pd
from fairseq_cli import train
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import options
from protein2def_dataset import *
from protein2def_model import *
from protein2def_task import *
from fairseq_cli import train, generate


def main():
    dataset = 'goa_human_cat'
    training_fold_range = [0]
    data_prefix = '../data/goa_human_terms_10_proteins_fairseq/databin_fold'
    for fold_i in training_fold_range:
        parser = options.get_generation_parser()
        src_maxL = 1000
        maxL = 1024
        parser.set_defaults(arch='transformer',
                            max_positions=(src_maxL, maxL),
                            max_tokens=12000,
                            path=data_prefix + '/checkpoints/checkpoint_last.pt',
                            beam=5,
                            max_len_b=200,
                            results_path=data_prefix + '/checkpoints')
        args = parser.parse_args()
        cfg = convert_namespace_to_omegaconf(args)
        cfg.task.data = data_prefix
        cfg.dataset.gen_subset='test'
        cfg.dataset.skip_invalid_size_inputs_valid_test = True
        generate.main(cfg)


if __name__ == '__main__':
    main()