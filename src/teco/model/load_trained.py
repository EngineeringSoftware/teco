import pytorch_lightning as pl
import seutil as su


def load_trained(model_cls: str, ckpt_path: su.arg.RPath) -> pl.LightningModule:
    if model_cls == "CodeT5":
        from teco.model.CodeT5 import CodeT5Module

        return CodeT5Module.load_from_checkpoint(ckpt_path, pretrained_tokenizer="_work/subtokenizer/codet5")
    elif model_cls == "CodeGPT":
        from teco.model.CodeGPT import CodeGPTModule

        return CodeGPTModule.load_from_checkpoint(ckpt_path, pretrained_tokenizer="_work/subtokenizer/codegpt")
    else:
        raise ValueError(f"Unknown model class: {model_cls}")
