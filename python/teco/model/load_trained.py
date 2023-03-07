from pathlib import Path
from typing import Union

import pytorch_lightning as pl
from jsonargparse.typing import Path_drw, Path_fr


def load_trained(
    model_cls: str, ckpt_path: Union[Path_fr, Path_drw, Path, str]
) -> pl.LightningModule:
    if isinstance(ckpt_path, (Path_fr, Path_drw)):
        ckpt_path = Path(ckpt_path.abs_path)

    if model_cls == "CodeT5":
        from teco.model.CodeT5 import CodeT5Module

        return CodeT5Module.load_from_checkpoint(ckpt_path)
    elif model_cls == "PretrainBimodalCodeT5":
        from teco.model.PretrainBimodalCodeT5 import PretrainBimodalCodeT5Module

        return PretrainBimodalCodeT5Module.load_from_checkpoint(ckpt_path)
    elif model_cls == "PretrainDenoiseCodeT5":
        from teco.model.PretrainDenoiseCodeT5 import PretrainDenoiseCodeT5Module

        return PretrainDenoiseCodeT5Module.load_from_checkpoint(ckpt_path)
    elif model_cls == "CodeGPT":
        from teco.model.CodeGPT import CodeGPTModule

        return CodeGPTModule.load_from_checkpoint(ckpt_path)
    elif model_cls == "MultiTaskT5":
        from teco.model.MultiTaskT5 import MultiTaskT5Module

        return MultiTaskT5Module.load_from_checkpoint(ckpt_path)
    elif model_cls == "PrefixTuningCodeT5":
        from teco.model.PrefixTuningCodeT5 import PrefixTuningCodeT5Module

        return PrefixTuningCodeT5Module.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"Unknown model class: {model_cls}")
