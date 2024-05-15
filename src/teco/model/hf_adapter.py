import shlex
import sys
from typing import Optional

import pytorch_lightning as pl
import seutil as su
import torch
import transformers
from jsonargparse import CLI
from pytorch_lightning.utilities.cli import (
    LR_SCHEDULER_REGISTRY,
    OPTIMIZER_REGISTRY,
)

from teco.macros import Macros
from teco.model.utils import DefaultLightningCLI, PathSafeSaveConfigCallback


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return None


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: Optional[int] = None, val_batch_size: Optional[int] = None):
        super().__init__()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(DummyDataset())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(DummyDataset())


class HuggingFaceAdapter:
    def push(self, model_cls: str, ckpt_path: su.arg.RPath, repo_id: str):
        if model_cls == "CodeT5":
            from teco.model.CodeT5 import CodeT5Module

            pl_module = CodeT5Module.load_from_checkpoint(ckpt_path, pretrained_tokenizer="_work/subtokenizer/codet5")
        elif model_cls == "CodeGPT":
            from teco.model.CodeGPT import CodeGPTModule

            pl_module = CodeGPTModule.load_from_checkpoint(ckpt_path, pretrained_tokenizer="_work/subtokenizer/codegpt")
        else:
            raise ValueError(f"Unknown model class: {model_cls}")

        pl_module.tokenizer.tokenizer.push_to_hub(repo_id)
        pl_module.model.push_to_hub(repo_id)

    def pull(self, model_cls: str, exp_dir: su.arg.RPath, repo_id: str, args: Optional[str] = None):
        su.io.rmdir(exp_dir)
        ckpt_path = exp_dir / "model" / "last.ckpt"

        if model_cls == "CodeT5":
            from teco.model.CodeT5 import CodeT5Module

            pl_module_cls = CodeT5Module
            config = "configs/CodeT5.yaml"
            pretrained_tokenizer = "_work/subtokenizer/codet5"
        elif model_cls == "CodeGPT":
            from teco.model.CodeGPT import CodeGPTModule

            pl_module_cls = CodeGPTModule
            config = "configs/CodeGPT.yaml"
            pretrained_tokenizer = "_work/subtokenizer/codegpt"
        else:
            raise ValueError(f"Unknown model class: {model_cls}")

        OPTIMIZER_REGISTRY.register_classes(transformers.optimization, torch.optim.Optimizer, override=True)
        LR_SCHEDULER_REGISTRY.register_classes(
            transformers.optimization, torch.optim.lr_scheduler._LRScheduler, override=True
        )

        with su.io.cd(Macros.project_dir):
            sys.argv = [
                sys.argv[0],
                "--config",
                config,
                "--exp_dir",
                str(exp_dir.absolute()),
                "--model.pretrained_tokenizer",
                pretrained_tokenizer,
                "--model.pretrained_model",
                repo_id,
            ]
            if args is not None:
                sys.argv += shlex.split(args)
            cli = DefaultLightningCLI(
                pl_module_cls,
                DummyDataModule,
                save_config_callback=PathSafeSaveConfigCallback,
                optimizers=[(None, "optimizer", "model.optimizer_init")],
                lr_schedulers=[(None, "lr_scheduler", "model.lr_scheduler_init")],
                run=False,
            )
            try:
                cli.trainer.fit(cli.model, torch.utils.data.DataLoader(DummyDataset()))
            except TypeError:
                pass
            cli.trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":
    CLI(HuggingFaceAdapter, as_positional=False)
