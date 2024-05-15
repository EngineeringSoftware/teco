import atexit
import collections
import concurrent
import dataclasses
import datetime
import functools
import json
import operator
import os
import time
import traceback
from pathlib import Path, PosixPath
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import pytorch_lightning as pl
import seutil as su
import torch
import yaml
from jsonargparse.typing import Path_dc, Path_drw, Path_dw, Path_fc, Path_fr
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.cli import (
    LR_SCHEDULER_REGISTRY,
    OPTIMIZER_REGISTRY,
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from seutil.MiscUtils import itos_human_readable
from tqdm import tqdm

logger = su.log.get_logger(__name__)


class PathSafeSaveConfigCallback(SaveConfigCallback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = self.remove_cwd(self.config)

    @classmethod
    def remove_cwd(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        ret = {}
        for k, v in config.items():
            if isinstance(v, Path):
                v = os.path.relpath(v, Path.cwd())
            elif isinstance(v, (Path_drw, Path_dc, Path_dw, Path_fr, Path_fc)):
                v = os.path.relpath(Path(v.abs_path), Path.cwd())
            elif isinstance(v, str):
                v = v.replace(str(Path.cwd()), "")
            elif isinstance(v, dict):
                v = cls.remove_cwd(v)
            ret[k] = v
        return ret


class BestAndLastModelCheckpoint(ModelCheckpoint):
    """
    A ModelCheckpoint callback that only saves the best and last checkpoints.
    The last checkpoint will be named as "last.ckpt" and include everything.
    The best checkpoint (at most one) will be named as "best.ckpt" and only include the model state.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: bool = True,
        save_best: bool = True,
        mode: str = "min",
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[datetime.timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
    ):
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=1 if save_best else 0,
            save_weights_only=True,
            mode=mode,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )

    def format_checkpoint_name(
        self, metrics: dict, filename: Optional[str] = None, ver: Optional[int] = None
    ) -> str:
        # hacked to only handle best/last checkpoints
        if filename is None:
            ckpt_name = "best.ckpt"
        else:
            ckpt_name = "last.ckpt"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name

    def _save_last_checkpoint(
        self, trainer: "pl.Trainer", monitor_candidates: dict
    ) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, "last")
        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        trainer.save_checkpoint(filepath, False)
        if previous and previous != filepath:
            trainer.strategy.remove_checkpoint(previous)


class DefaultLightningCLI(LightningCLI):
    def __init__(
        self,
        *args,
        prediction_writer: Optional[Callback] = None,
        optimizers: Optional[
            List[Tuple[Optional[Union[Type, List[Type]]], str, str]]
        ] = None,
        lr_schedulers: Optional[
            List[Tuple[Optional[Union[Type, List[Type]]], str, str]]
        ] = None,
        **kwargs,
    ):
        self.prediction_writer = prediction_writer
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        kwargs.setdefault("save_config_overwrite", True)
        super().__init__(*args, **kwargs)
        self.ckpt_dir: Path = None

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--exp_dir",
            required=False,
            help="[fit/predict] Path to the experiment directory",
            type=Union[Path_drw, Path_dc],
        )

        parser.add_argument(
            "--resume",
            required=False,
            help="[fit] What to do if a checkpoint already exists: unset (default) = error; True = resume from the latest/selected checkpoint; False = remove all existing checkpoints",
            type=bool,
        )

        parser.add_argument(
            "--no_train",
            required=False,
            help="[fit] Do not train the model, but only save a last.ckpt",
            type=bool,
            default=False,
        )

        parser.add_argument(
            "--ckpt_name",
            required=False,
            help="[fit/predict] The checkpoint file name to load (under regular ckpt directory); if unset, the latest checkpoint will be loaded",
            type=str,
        )

        parser.add_argument(
            "--no_ckpt_ok",
            required=False,
            help="[predict] What to do if no checkpoint exists: False (default) = halt; True = initialize a fresh module",
            type=bool,
            default=False,
        )

        parser.add_argument(
            "--output_dir",
            required=False,
            help="[predict] Path to the output directory",
            type=Path_dc,
        )

        parser.add_lightning_class_args(BestAndLastModelCheckpoint, "ckpt")
        parser.set_defaults(
            {
                "ckpt.save_last": True,
                "ckpt.save_best": False,
                "ckpt.verbose": True,
                "ckpt.monitor": "loss/val",
                "ckpt.mode": "min",
                "ckpt.every_n_epochs": 1,
            }
        )

        if self.optimizers is not None:
            for types, nested_key, link_to in self.optimizers:
                # if types is None:
                #     types = OPTIMIZER_REGISTRY.classes
                parser.add_optimizer_args((torch.optim.Optimizer,), nested_key, link_to)

        if self.lr_schedulers is not None:
            for types, nested_key, link_to in self.lr_schedulers:
                # if types is None:
                #     types = LR_SCHEDULER_REGISTRY.classes
                parser.add_lr_scheduler_args(
                    pl.cli.LRSchedulerTypeTuple, nested_key, link_to
                )

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()

        if "subcommand" not in self.config:
            subcommand = "unknown"
            config = self.config
        else:
            subcommand = self.config["subcommand"]
            config = self.config[self.config["subcommand"]]

        # Set up experiment directory
        if config["exp_dir"] is not None:
            exp_dir = Path(config["exp_dir"].abs_path).resolve()
            su.io.mkdir(exp_dir)
        else:
            if subcommand == "fit":
                raise MisconfigurationException("exp_dir is required for training")
            exp_dir = su.io.mktmp_dir("pl")
            logger.info(
                f"Using temporary directory {exp_dir} as exp_dir, which will be deleted after the run"
            )
            atexit.register(lambda: su.io.rmdir(exp_dir))
        config["trainer"]["default_root_dir"] = os.path.relpath(exp_dir, Path.cwd())
        ckpt_dir = exp_dir / "model"
        self.ckpt_dir = ckpt_dir
        su.io.mkdir(ckpt_dir)
        config["ckpt"]["dirpath"] = os.path.relpath(ckpt_dir, Path.cwd())

        # In ddp mode, default disable find_unused_parameters
        if config["trainer"]["strategy"] == "ddp":
            config["trainer"]["strategy"] = pl.strategies.DDPStrategy(
                find_unused_parameters=False,
            )
            # this doesn't pass typechecking
            # {
            #     "class_path": "pytorch_lightning.plugins.DDPPlugin",
            #     "init_args": {"find_unused_parameters": False},
            # }
        elif config["trainer"]["strategy"] == "ddp_spawn":
            config["trainer"]["strategy"] = pl.strategies.DDPSpawnStrategy(
                find_unused_parameters=False,
            )

        # Locate checkpoint, if there is one
        if config.get("ckpt_path") is None:
            if config.get("ckpt_name") is not None:
                ckpt_file = ckpt_dir / config["ckpt_name"]
            else:
                ckpt_file = self.locate_ckpt(ckpt_dir)
        else:
            if isinstance(config["ckpt_path"], Path_drw):
                ckpt_file = Path(config["ckpt_path"].abs_path).resolve()
            else:
                ckpt_file = Path(config["ckpt_path"]).resolve()

        if subcommand == "fit":
            # If a checkpoint path is specified, assume we want to resume from it
            if config["ckpt_path"] is not None or config["ckpt_name"] is not None:
                if config["resume"] is None:
                    config["resume"] = True

            # If there is a checkpoint, we must decide what to do with it
            if ckpt_file is not None:
                if config["resume"] is None:
                    raise MisconfigurationException(
                        f"A checkpoint is present at {ckpt_file}, but I'm not sure what to do with it. Either set `--resume True` to use it or `--resume False` to overwrite it."
                    )
                elif config["resume"] is True:
                    logger.info(f"Resuming from checkpoint {ckpt_file}")
                    config["ckpt_path"] = str(ckpt_file.resolve())
                else:
                    logger.info(f"Removing checkpoints under {ckpt_dir}")
                    su.io.mkdir(ckpt_dir, fresh=True)
                    config["ckpt_path"] = None
        elif subcommand == "predict":
            # Set up checkpoint to load
            if ckpt_file is not None:
                config["ckpt_path"] = str(ckpt_file.resolve())
            else:
                if config["no_ckpt_ok"] is False:
                    raise MisconfigurationException(
                        f"No checkpoint found, cannot predict (unless using `--no_ckpt_ok True` to allow predicting from scratch)"
                    )
                else:
                    logger.info("No checkpoint found, predicting from scratch")

            # Set up prediction writer
            if self.prediction_writer is None:
                logger.warning(
                    "No prediction writer specified. "
                    "Will not write predictions to disk."
                )
            elif config["output_dir"] is None:
                logger.warning(
                    "No output directory specified."
                    "Will not write predictions to disk."
                )
            else:
                if config["trainer"]["callbacks"] is None:
                    config["trainer"]["callbacks"] = []
                config["trainer"]["callbacks"].append(
                    self.prediction_writer(config["output_dir"])
                )

        # Set up logger
        logger_save_dir = exp_dir / "logs" / subcommand
        logger_version = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        while (logger_save_dir / logger_version).exists():
            time.sleep(1)
            logger_version = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        su.io.mkdir(logger_save_dir)
        config["trainer"]["logger"] = {
            "class_path": "pytorch_lightning.loggers.tensorboard.TensorBoardLogger",
            "init_args": {
                "save_dir": os.path.relpath(logger_save_dir, Path.cwd()),
                "name": None,
                "version": logger_version,
            },
        }

    def after_fit(self):
        config = self.config[self.config["subcommand"]]

        # if using deepspeed, clean up old model checkpoints
        strategy = config["trainer"]["strategy"]
        if (
            strategy is not None
            and isinstance(strategy, str)
            and strategy.startswith("deepspeed")
        ):
            exp_dir = Path(config["exp_dir"].abs_path).resolve()
            save_dir = exp_dir / "model" / "last.ckpt"
            latest = su.io.load(save_dir / "latest", su.io.Fmt.txt)
            to_remove = []
            for d in save_dir.glob("*"):
                if d.is_dir() and d.name != latest:
                    to_remove.append(d)
            for d in to_remove:
                su.io.rmdir(d)

    @classmethod
    def locate_ckpt(cls, ckpt_dir: Optional[Path]) -> Optional[Path]:
        if ckpt_dir is None:
            return None

        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if len(ckpt_files) == 0:
            ckpt_file = None
            logger.info(f"No checkpoint found in {ckpt_dir}")
        elif len(ckpt_files) == 1:
            ckpt_file = ckpt_files[0]
            logger.info(f"Found one checkpoint in {ckpt_dir}: {ckpt_file.name}")
        else:
            if (ckpt_dir / "last.ckpt").is_file():
                ckpt_file = ckpt_dir / "last.ckpt"
                logger.info(
                    f"Found the last checkpoint in {ckpt_dir}: {ckpt_file.name}"
                )
            else:
                ckpt_file = sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]
                logger.warning(
                    f"Multiple checkpoints found in {ckpt_dir}: {[x.name for x in ckpt_files]}; picking the latest modified: {ckpt_file.name}"
                )
        return ckpt_file


class SimpleDataset(torch.utils.data.Dataset):
    """
    A simple map-style dataset created from a sequence of things.
    """

    def __init__(self, seq: Sequence[Any]):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return self.seq[idx]


class ProcessingOnTheFlyDataset(torch.utils.data.Dataset):
    """
    This dataset applies a process function on input data on the fly.
    The results can be optionally cached, improving speed but consuming memory.
    """

    def __init__(
        self,
        process_fn: Callable,
        *args,
        cached: bool = False,
        debug: bool = False,
        non_iter_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if non_iter_kwargs is None:
            non_iter_kwargs = {}

        self.process_fn = process_fn
        self.args = args
        self.kwargs = kwargs
        self.non_iter_kwargs = non_iter_kwargs
        self.cached = cached
        if self.cached:
            self.cache = {}
        self.debug = debug

        # check size
        self.len = None
        for arg in self.args:
            if arg is not None:
                if self.len is None:
                    self.len = len(arg)
                else:
                    assert self.len == len(arg), "all inputs must have equal length"
        for k, v in self.kwargs.items():
            if v is not None:
                if self.len is None:
                    self.len = len(v)
                else:
                    assert self.len == len(v), "all inputs must have equal length"
        assert self.len is not None, "at least one non-None input is required"

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.cached and idx in self.cache:
            return self.cache[idx]

        try:
            ret = self.process_fn(
                *[arg[idx] for arg in self.args],
                **{k: v[idx] for k, v in self.kwargs.items()},
                **self.non_iter_kwargs,
            )
        except Exception as e:
            logger.error(
                f"Error during processing item {idx}: {traceback.format_exc()}"
            )
            raise RuntimeError(e)

        if self.cached:
            self.cache[idx] = ret
        if self.debug:
            print(f"getitem {idx}/{self.len}:")
            for i, v in enumerate(ret):
                if isinstance(v, torch.Tensor):
                    print(f"  {i}: tensor of shape {v.shape}")
                else:
                    print(f"  {i}: type {type(v)}")
        return ret


class LoadingAndProcessingOnTheFlyDataset(torch.utils.data.Dataset):
    """
    This dataset loads the data and applies a process function on the fly.
    To enable random access later, it go through all data files to find line ending positions during __init__. To avoid repeated work, it is recommended to init this dataset in the main process and then distribute to multiple processes.
    """

    def __init__(
        self,
        save_dir: Path,
        clz: type,
        process_fn: Callable,
        *args,
        only: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        expected_ids: Set[int] = None,
        pbar: Optional[tqdm] = None,
        non_iter_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if not save_dir.is_dir():
            raise FileNotFoundError(f"Not found saved data at {save_dir}")
        self.save_dir = save_dir
        self.clz = clz
        self.process_fn = process_fn
        self.args = args
        self.kwargs = kwargs
        self.non_iter_kwargs = non_iter_kwargs

        # figure out the fields to load
        fields: Dict[str, dataclasses.Field] = {
            f.name: f for f in dataclasses.fields(clz)
        }
        if "id" not in fields or "fully_deserialized" not in fields:
            raise ValueError(
                f"Not a proper data class: {clz}; need to have id and fully_deserialized fields"
            )
        del fields["id"]
        del fields["fully_deserialized"]

        if only is not None:
            fields = {n: f for n, f in fields.items() if n in only}
        if exclude is not None:
            fields = {n: f for n, f in fields.items() if n not in exclude}

        # open the relevant files
        f2file = {}
        drop_fields = []
        self.f2type = {}
        for name, field in fields.items():
            if not (save_dir / f"{name}.jsonl").is_file():
                if only is not None:
                    raise FileNotFoundError(f"Not found {name} in {only} in {save_dir}")
                else:
                    logger.warning(
                        f"Not found {name}.jsonl at {save_dir}, this field won't be loaded"
                    )
                    drop_fields.append(name)
                    continue

            if (
                isinstance(field, dataclasses.Field)
                and field.metadata.get("shared") is True
            ):
                logger.warning(
                    f"Seeing an shared field {name}, not sure how to handle it yet"
                )
                drop_fields.append(name)
                continue
            f2file[name] = open(save_dir / f"{name}.jsonl", "r")
            self.f2type[name] = field.type
        for f in drop_fields:
            del fields[f]

        # load all ids
        all_ids = su.io.load(save_dir / "id.jsonl")

        # read all files to memorize line ending positions
        self.ids = []
        self.f2lines = collections.defaultdict(list)
        if pbar is not None:
            pbar.reset(len(all_ids))
        for i in all_ids:
            if expected_ids is not None and i not in expected_ids:
                # skip this line in all files
                for f in f2file:
                    f2file[f].readline()
            else:
                # record the id
                self.ids.append(i)

                # record the current file position, and proceed one line
                for f in f2file:
                    self.f2lines[f].append(f2file[f].tell())
                    f2file[f].readline()

            pbar.update(1)

        # close all files
        for file in f2file.values():
            file.close()

        self.f2file = None

        # TODO: sizes of args, kwargs are not checked yet

    def open_files(self):
        if self.f2file is None:
            self.f2file = {
                f: open(self.save_dir / f"{f}.jsonl", "r") for f in self.f2type
            }

    def close_files(self):
        if self.f2file is not None:
            for file in self.f2file.values():
                file.close()
            self.f2file = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # make sure files are open
        self.open_files()

        try:
            f2value = {"id": self.ids[idx]}

            # move files to correct position and read line
            for f in self.f2file:
                self.f2file[f].seek(self.f2lines[f][idx])
                f2value[f] = su.io.deserialize(
                    json.loads(self.f2file[f].readline()), self.f2type[f]
                )

            # assemble data
            data = self.clz(**f2value)

            # process data
            ret = self.process_fn(
                data,
                *[arg[idx] for arg in self.args],
                **{k: v[idx] for k, v in self.kwargs.items()},
                **(self.non_iter_kwargs or {}),
            )
        except Exception as e:
            logger.error(
                f"Error during processing item {idx}: {traceback.format_exc()}"
            )
            raise RuntimeError(e)

        return ret


def shape_sequences(
    seqs: list,
    pad: Any,
    shapes: Sequence[Optional[int]] = None,
    pad_sides: Sequence[Optional[str]] = None,
    truncate_sides: Sequence[Optional[str]] = None,
) -> list:
    """
    Fits the multi-dimentional sequences into the given shape.

    :param seqs: the multi-dimentional sequences
    :param pad: the padding value to be applied at the last dimension
    :param shape: the shape of each dimension; a None value means no size constrains for that dimension
    :param pad_side: the side to which the padding is applied, either "left" or "right"; "None" can be used only if there is no size constrains for that dimension
    :param truncate_side: the side from which the truncation is applied, either "left" or "right"; "None" can be used only if there is no size constrains for that dimension

    :return: the padded and truncated multi-dimentional sequences; if the shapes on all dimensions (except for the first dimension) are non-None, the return value is directly convertable to a tensor
    """
    if pad_sides is None:
        pad_sides = ["right"] * len(shapes)
    if truncate_sides is None:
        truncate_sides = ["right"] * len(shapes)
    if not (len(shapes) == len(pad_sides) == len(truncate_sides)):
        raise ValueError(
            f"dimensions of shapes({len(shapes)}), pad_sides({len(pad_sides)}), and truncate_sides({len(truncate_sides)}) must match"
        )

    if len(shapes) < 1:
        raise ValueError("need at least 1d data")
    elif len(shapes) == 1:
        shape = shapes[0]
        pad_side = pad_sides[0]
        truncate_side = truncate_sides[0]

        if shape is None:
            return seqs
        else:
            if truncate_side == "right":
                seqs = seqs[:shape]
            elif truncate_side == "left":
                seqs = seqs[-shape:]
            else:
                raise ValueError(f"truncate_side must be one of 'left' or 'right'")

            missing = shape - len(seqs)
            if missing > 0:
                if pad_side == "right":
                    seqs = seqs + [pad] * missing
                elif pad_side == "left":
                    seqs = [pad] * missing + seqs
                else:
                    raise ValueError(f"pad_side must be one of 'left' or 'right'")
            return seqs
    else:
        shape = shapes[0]
        pad_side = pad_sides[0]
        truncate_side = truncate_sides[0]

        if shape is not None:
            if truncate_side == "right":
                seqs = seqs[:shape]
            elif truncate_side == "left":
                seqs = seqs[-shape:]
            else:
                raise ValueError(f"truncate_side must be one of 'left' or 'right'")

            missing = shape - len(seqs)
            if missing > 0:
                if pad_side == "right":
                    seqs = seqs + [[]] * missing
                elif pad_side == "left":
                    seqs = [[]] * missing + seqs
                else:
                    raise ValueError(f"pad_side must be one of 'left' or 'right'")

        return [
            shape_sequences(s, pad, shapes[1:], pad_sides[1:], truncate_sides[1:])
            for s in seqs
        ]


def apply_1d_sequences(
    seqs: list, func: Callable[[list], list], *args, **kwargs
) -> list:
    """
    Applies a function to each 1d sub-sequence of the multi-dimentional sequences.

    :param seqs: the multi-dimentional sequences
    :param func: the function to apply
    :param args: the arguments to pass to the function
    :param kwargs: the keyword arguments to pass to the function

    :return: the sequences after applying the function
    """
    if not isinstance(seqs, list):
        raise ValueError(f"seqs must be a list, but got {type(seqs)}")

    if len(seqs) == 0 or not isinstance(seqs[0], list):
        return func(seqs, *args, **kwargs)
    else:
        return [apply_1d_sequences(s, func, *args, **kwargs) for s in seqs]


def apply_elem(seqs: list, func: Callable[[Any], Any], *args, **kwargs) -> list:
    """
    Applies a function to each element of the multi-dimentional sequences.

    :param seqs: the multi-dimentional sequences
    :param func: the function to apply
    :param args: the arguments to pass to the function
    :param kwargs: the keyword arguments to pass to the function

    :return: the sequences after applying the function
    """
    if not isinstance(seqs, list):
        raise ValueError(f"seqs must be a list, but got {type(seqs)}")

    if len(seqs) == 0 or not isinstance(seqs[0], list):
        return [func(s, *args, **kwargs) for s in seqs]
    else:
        return [apply_elem(s, func, *args, **kwargs) for s in seqs]


def parallel_map(
    func: Callable,
    *iterables,
    pbar: Optional[tqdm] = True,
    max_workers: Optional[int] = None,
    use_mp: bool = True,
    timeout: Optional[float] = None,
):
    ret = []
    if use_mp:
        executor_cls = concurrent.futures.ThreadPoolExecutor
    else:
        executor_cls = concurrent.futures.ProcessPoolExecutor

    with executor_cls(max_workers=max_workers) as executor:
        futures = []
        for inp in zip(*iterables):
            future = executor.submit(func, *inp)
            if pbar is not None:
                future.add_done_callback(lambda p: pbar.update(1))
            futures.append(future)

        for future in futures:
            ret.append(future.result(timeout=timeout))
    return ret


def fmt_gpu_mem_info(gpu_id=0, brief=True) -> str:
    import torch.cuda.memory

    if torch.cuda.is_available():
        report = ""
        t = torch.cuda.get_device_properties(gpu_id).total_memory
        c = torch.cuda.memory.memory_reserved(gpu_id)
        a = torch.cuda.memory_allocated(gpu_id)
        f = t - a

        t = itos_human_readable(t).replace("B", "G") + "B"
        c = itos_human_readable(c).replace("B", "G") + "B"
        a = itos_human_readable(a).replace("B", "G") + "B"
        f = itos_human_readable(f).replace("B", "G") + "B"

        report += f"[Allocated {a} | Free {f} | Cached {c} | Total {t}]\n"
        if not brief:
            report += torch.cuda.memory_summary(device=gpu_id, abbreviated=True)
        return report
    else:
        return f"CUDA not available, using CPU"


def fmt_all_tensors_mem_info(gpu_only: bool = False, max_lines: int = 30) -> str:
    import gc

    import torch

    report = ""

    id2tensor = {}

    def add_tensor(obj):
        if torch.is_tensor(obj):
            tensor = obj
        elif hasattr(obj, "data") and torch.is_tensor(obj.data):
            tensor = obj.data
        else:
            return

        if (gpu_only and tensor.is_cuda) or not gpu_only:
            id2tensor[id(tensor)] = tensor

    for obj in gc.get_objects():
        try:
            # Add the obj if it is a tensor.
            add_tensor(obj)
            # Some tensors are "saved & hidden" for the backward pass.
            # TODO: this hack doesn't capture all memory used for backward pass
            if hasattr(obj, "saved_tensors"):
                for tensor_obj in obj.saved_tensors:
                    add_tensor(tensor_obj)
        except KeyboardInterrupt:
            raise
        except:
            pass

    id2size = {}
    for i, t in id2tensor.items():
        if len(t.size()) == 0:
            s = 0
        else:
            s = functools.reduce(operator.mul, t.size())

        id2size[i] = s

    report += (
        f"Total tensors: {len(id2tensor)}; Total size: {sum(id2size.values()):,d}\n"
    )
    counter = collections.Counter()

    for i in sorted(id2tensor.keys(), key=lambda x: id2size[x], reverse=True):
        t = id2tensor[i]
        s = id2size[i]
        counter[f"{t.type():>24} {s:>12,d} ({t.size()})"] += 1

    count_printed = 0
    for desc, c in counter.most_common(max_lines):
        report += f"{c:>3,d}* {desc}\n"
        count_printed += c
    if count_printed < len(id2tensor):
        report += f"  ({len(id2tensor) - count_printed:>3,d} more tensors omitted...)\n"

    return report
