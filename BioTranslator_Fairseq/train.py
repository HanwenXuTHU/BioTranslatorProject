from tasks import protein_zero_shot
from tasks import cell_type_discovery
from models import biotranslator
from loss import bce_annotation_loss

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable
import numpy as np
import torch

sys.path.append(os.getcwd())

from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    for fold in range(cfg.task.k_fold):

        logger.info(
            "start cross-validation for fold {}".format(fold)
        )

        task.generate_fold_data(fold)

        # Build model and criterion
        if cfg.distributed_training.ddp_backend == "fully_sharded":
            with fsdp_enable_wrap(cfg.distributed_training):
                model = fsdp_wrap(task.build_model(cfg.model))
        else:
            model = task.build_model(cfg.model)
        criterion = task.build_criterion(cfg.criterion)
        logger.info(model)
        logger.info("task: {}".format(task.__class__.__name__))
        logger.info("model: {}".format(model.__class__.__name__))
        logger.info("criterion: {}".format(criterion.__class__.__name__))
        logger.info(
            "num. shared model params: {:,} (num. trained: {:,})".format(
                sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
                sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
            )
        )

        logger.info(
            "num. expert model params: {} (num. trained: {})".format(
                sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
                sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
            )
        )

        # (optionally) Configure quantization
        if cfg.common.quantization_config_path is not None:
            quantizer = quantization_utils.Quantizer(
                config_path=cfg.common.quantization_config_path,
                max_epoch=cfg.optimization.max_epoch,
                max_update=cfg.optimization.max_update,
            )
        else:
            quantizer = None

        # Build trainer
        if cfg.common.model_parallel_size == 1:
            trainer = Trainer(cfg, task, model, criterion, quantizer)
        else:
            trainer = MegatronTrainer(cfg, task, model, criterion)
        logger.info(
            "training on {} devices (GPUs/TPUs)".format(
                cfg.distributed_training.distributed_world_size
            )
        )
        logger.info(
            "max tokens per device = {} and max sentences per device = {}".format(
                cfg.dataset.max_tokens,
                cfg.dataset.batch_size,
            )
        )

        epoch_itr = trainer.get_train_iterator(epoch=1, load_dataset=True)
        # set the train embeddings
        criterion.set_text_emb(task.train_emb)

        max_epoch = cfg.optimization.max_epoch or math.inf
        lr = trainer.get_lr()
        train_meter = meters.StopwatchMeter()
        train_meter.start()
        while epoch_itr.next_epoch_idx <= max_epoch:
            if lr <= cfg.optimization.stop_min_lr:
                logger.info(
                    f"stopping training because current learning rate ({lr}) is smaller "
                    "than or equal to minimum learning rate "
                    f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
                )
                break

            # train for one epoch
            _, should_stop = train(cfg, trainer, task, epoch_itr)
            if should_stop:
                break

            task.evaluate(cfg, trainer, fold)

            epoch_itr = trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                # sharded data: get train iterator for next epoch
                load_dataset=False,
                # don't cache epoch iterators for sharded datasets
                disable_iterator_cache=True,
            )
        train_meter.stop()
        logger.info("done training in {:.1f} seconds".format(train_meter.sum))


@metrics.aggregate("train")
def train(
        cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)


    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
                "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return [], should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def cli_main(
        modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()

    # group = parser.add_argument_group('imbalanced', 'Imbalanced Sampler')
    # group.add_argument('--imbalanced', default=False, action='store_true')

    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
