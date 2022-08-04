import os
import torch

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.tasks import FairseqTask, register_task
import itertools
import json
import logging
import os
from argparse import Namespace
from protein2def_dataset import Protein2DefDataset

import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.tasks import LegacyFairseqTask, register_task


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def one_hot(seq, start=0, max_len=2000):
    AALETTER = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    AAINDEX = dict()
    for i in range(len(AALETTER)):
        AAINDEX[AALETTER[i]] = i + 1
    onehot = np.zeros((21, max_len), dtype=np.int32)
    l = min(max_len, len(seq))
    for i in range(start, start + l):
        onehot[AAINDEX.get(seq[i - start], 0), i] = 1
    onehot[0, 0:start] = 1
    onehot[0, start + l:] = 1
    return onehot


def load_prot2def_dataset(
    src_data_path,
    tgt_data_path,
    split,
    src,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    input_seqL,
    left_pad_target,
    prepend_bos=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, tgt_data_path):
            prefix = os.path.join(tgt_data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, tgt_data_path):
            prefix = os.path.join(tgt_data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, tgt_data_path)
                )
        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)
        if not combine:
            break

    if len(tgt_datasets) == 1:
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(tgt_datasets)
        sample_ratios[0] = upsample_primary
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(tgt_dict, "bos_index")
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    src_dataset = []
    src_lengths = []
    f = open(src_data_path.format(split))
    lines = f.readlines()
    for line in lines:
        seq_str = line.strip().split(' ')
        seq_str = ''.join(seq_str)
        src_dataset.append(one_hot(seq_str, max_len=input_seqL))
        src_lengths.append(input_seqL)
    return Protein2DefDataset(
        src=src_dataset,
        src_sizes=src_lengths,
        tgt=tgt_dataset,
        tgt_sizes=tgt_dataset_sizes,
        tgt_dict=tgt_dict,
        left_pad_target=left_pad_target,
        align_dataset=None,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple
    )



@register_task('protein2def_task')
class Protein2DefTask(LegacyFairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--tgt-data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--src-data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('-s', '--source-lang', default='en', metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default='de', metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--input-seqL', default=2000, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

    @classmethod
    def setup_task(cls, args, **kwargs):

        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        tgt_vocab = Dictionary.load(os.path.join(args.tgt_data, 'dict.de.txt'))
        print('| [Definition] dictionary: {} types'.format(len(tgt_vocab)))

        return Protein2DefTask(args, tgt_vocab)

    def __init__(self, args, tgt_vocab):
        super().__init__(args)
        self.tgt_dict = tgt_vocab

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        tgt_paths = utils.split_paths(self.args.tgt_data)
        assert len(tgt_paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            tgt_paths = tgt_paths[:1]
        tgt_data_path = tgt_paths[(epoch - 1) % len(tgt_paths)]

        src_data_path = self.args.src_data.format(split)
        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_prot2def_dataset(
            src_data_path,
            tgt_data_path,
            split,
            src,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            input_seqL=self.args.input_seqL,
            left_pad_target=self.args.left_pad_target,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return Protein2DefDataset(
            src_tokens,
            src_lengths,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.input_seqL, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

