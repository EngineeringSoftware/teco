import collections
import os
import random
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import seutil as su
import torch.utils.data
import transformers
from jsonargparse.typing import Path_drw
from pytorch_lightning.utilities.cli import LR_SCHEDULER_REGISTRY, OPTIMIZER_REGISTRY
from teco.macros import Macros
from teco.model.CodeT5 import CodeT5Module
from teco.model.subtokenizer_bpe import SubtokenizerBPE
from teco.model.utils import DefaultLightningCLI, PathSafeSaveConfigCallback

logger = su.log.get_logger(__name__)


class CodeT5NoFinetuneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: Optional[Path_drw] = None,
        val_data_dir: Optional[Path_drw] = None,
        batch_size: int = 1,
        val_batch_size: int = 8,
        max_seq_len: int = 512,
    ):
        super().__init__()
        if isinstance(train_data_dir, Path_drw):
            train_data_dir = os.path.relpath(Path(train_data_dir.abs_path), Path.cwd())
        if isinstance(val_data_dir, Path_drw):
            val_data_dir = os.path.relpath(Path(val_data_dir.abs_path), Path.cwd())
        self.save_hyperparameters()

        self.tokenizer: SubtokenizerBPE = None
        self.train_dataset = None

    def prepare_data(self):
        self.tokenizer = self.trainer.lightning_module.tokenizer
        self.hparams.inputs = self.trainer.lightning_module.hparams.inputs
        self.hparams.output = self.trainer.lightning_module.hparams.output
        pass

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = self.trainer.lightning_module.tokenizer
        self.hparams.inputs = self.trainer.lightning_module.hparams.inputs
        self.hparams.output = self.trainer.lightning_module.hparams.output
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.tensor([0, 1]),
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            # persistent_workers=True,
            # pin_memory=True,  # Not working on multi gpu
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.tensor([0, 1]),
            shuffle=False,
            batch_size=self.hparams.val_batch_size,
            num_workers=0,
            # persistent_workers=True,
            # pin_memory=True,  # Not working on multi gpu
        )


# class CodeT5NoFinetuneModule(CodeT5Module):
#     def __init__(
#         self,
#         pretrained_tokenizer: Union[Path_drw, str],
#         pretrained_model: Union[Path_drw, str],
#         optimizer_init: dict,
#         lr_scheduler_init: dict,
#         inputs: List[Input] = [Input.focalm, Input.sign, Input.prev_stmts],
#         output: Output = Output.stmt,
#         loss_scale_step: int = -1,  # recommended 5000
#         loss_scale_important: float = 2.0,
#         loss_scale_unimportant: float = 0.2,
#     ):
#         super().__init__(
#             pretrained_tokenizer,
#             pretrained_model,
#             optimizer_init,
#             lr_scheduler_init,
#             inputs,
#             output,
#             loss_scale_step,
#             loss_scale_important,
#             loss_scale_unimportant,
#         )

#     def training_step(
#         self,
#         batch: List[torch.Tensor],
#         batch_idx: int = -1,
#         data_loader_idx: Optional[int] = None,
#     ) -> torch.Tensor:
#         raise StopIteration()
#         pass

#     def training_epoch_end(self, outputs):
#         pass

#     def validation_step(
#         self,
#         batch: List[torch.Tensor],
#         batch_idx: int = -1,
#         data_loader_idx: Optional[int] = None,
#     ):
#         pass

#     def validation_epoch_end(self, outputs):
#         pass


if __name__ == "__main__":
    su.log.setup(Macros.log_file)

    OPTIMIZER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.Optimizer, override=True
    )
    LR_SCHEDULER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.lr_scheduler._LRScheduler, override=True
    )

    def modified_training_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int = -1,
        data_loader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        raise StopIteration()

    CodeT5Module.training_step = modified_training_step
    CodeT5Module.validation_step = lambda self, *args, **kwargs: None
    CodeT5Module.validation_epoch_end = lambda self, *args, **kwargs: self.log_dict(
        {"val/xmatch": 0}
    )

    cli = DefaultLightningCLI(
        CodeT5Module,
        CodeT5NoFinetuneDataModule,
        save_config_callback=PathSafeSaveConfigCallback,
        optimizers=[(None, "optimizer", "model.optimizer_init")],
        lr_schedulers=[(None, "lr_scheduler", "model.lr_scheduler_init")],
    )
    cli.trainer.save_checkpoint(
        Path(cli.config["fit"]["ckpt"]["dirpath"]) / "last.ckpt"
    )
