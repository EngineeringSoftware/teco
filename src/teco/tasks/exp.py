import os
import shlex
from pathlib import Path
from typing import Dict, Optional, Tuple

import seutil as su
from invoke import task
from teco.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)


@task
def subtokenizer_codet5(c):
    su.io.mkdir(Macros.work_dir / "subtokenizer")
    c.run(
        f"""python -m teco.model.subtokenizer_bpe\
            --base=Salesforce/codet5-base\
            --out_dir={Macros.work_dir}/subtokenizer/codet5 """
    )


@task
def subtokenizer_codet5_vanilla(c):
    su.io.mkdir(Macros.work_dir / "subtokenizer")
    c.run(
        f"""python -m teco.model.subtokenizer_bpe\
            --base=Salesforce/codet5-base\
            --out_dir={Macros.work_dir}/subtokenizer/codet5-vanilla\
            --add_simp_insn_tokens=False\
            --add_insn_tokens=False """
    )


DECODING_VARIANTS: Dict[str, Tuple[str, dict]] = {
    # greedy
    "greedy": ("greedy", {}),
    "greedy_t": ("greedy", {"no_repeat_ngram": 2}),
    # sampling
    "topp09": ("sampling", {"top_p": 0.9}),
    "temp08": ("sampling", {"temperature": 0.8}),
    "sampling1": (
        "sampling",
        {"top_p": 0.9, "temperature": 0.8, "num_return_sequences": 1},
    ),
    "sampling2": (
        "sampling",
        {"top_p": 0.9, "temperature": 0.8, "num_return_sequences": 2},
    ),
    "sampling10": (
        "sampling",
        {"top_p": 0.9, "temperature": 0.8, "num_return_sequences": 10},
    ),
    "sampling100": (
        "sampling",
        {"top_p": 0.9, "temperature": 0.8, "num_return_sequences": 100},
    ),
    # beam search
    "bs10": (
        "beam_search",
        {"beam_size": 10, "num_return_sequences": 10, "repetition_penalty": 5.0},
    ),
}


@task
def exp_train(
    c,
    setup: str,
    exp: str,
    model: Optional[str] = None,
    config_name: Optional[str] = None,
    overwrite: bool = False,
    use_base: Optional[str] = None,
    subtokenizer: str = "codet5",
    args: str = "",
    train_subset: str = "train/train",
    val_subset: str = "train/val",
    exp_group: Optional[str] = None,
    local: bool = False,
    slurm: Optional[int] = None,
):
    # default model name is the first part of exp name
    if model is None:
        model = exp.split("-")[0]
    # default config name follows the model name
    if config_name is None:
        config_name = f"{model}.yaml"

    if local:
        args = "--config configs/local.yaml " + args

    if exp_group is not None:
        exp_dir = Macros.work_dir / "exp" / setup / "train" / exp_group / exp
    else:
        exp_dir = Macros.work_dir / "exp" / setup / "train" / exp

    if overwrite:
        su.io.mkdir(exp_dir, fresh=True)
    else:
        su.io.mkdir(exp_dir)

    s = f"python -m teco.model.{model} fit"
    s += f" --exp_dir {exp_dir}"
    s += f" --config configs/{config_name}"
    s += f" --data.train_data_dir {Macros.work_dir}/setup/{setup}/{train_subset}"
    s += f" --data.val_data_dir {Macros.work_dir}/setup/{setup}/{val_subset}"
    s += f" --model.pretrained_tokenizer {Macros.work_dir}/subtokenizer/{subtokenizer}"
    if use_base is not None:
        s += f" --model.pretrained_model {use_base}"
    if len(args) > 0:
        s += f" {args}"

    # distributed training
    if slurm is not None:
        ngpus = 4
        master_addr = os.environ["HOSTNAME"]
        master_port = 12910
        world_size = slurm * ngpus
        srun_args = f"--nodes={slurm} --ntasks={slurm} --ntasks-per-node=1"
        s = f"""
            srun {srun_args}\
            inv exp.slurm-helper\
            --master-addr={master_addr}\
            --master-port={master_port}\
            --world-size={world_size}\
            --script={shlex.quote(s)} """
    c.run(s)


@task
def slurm_helper(
    c,
    master_addr: str,
    master_port: int,
    world_size: int,
    script: str,
):
    c.run(
        f"MASTER_ADDR={master_addr} MASTER_PORT={master_port} WORLD_SIZE={world_size} NODE_RANK=$SLURM_NODEID {script}"
    )


@task
def exp_eval(
    c,
    setup: str,
    exp: str,
    eval_set: str,
    model: Optional[str] = None,
    config_name: Optional[str] = None,
    overwrite: bool = False,
    no_ckpt_ok: bool = True,
    subtokenizer: Optional[str] = None,
    eval_method: str = "default",
    output_dir: Optional[Path] = None,
    args: str = "",
    local: bool = False,
):
    # default model name is the first part of exp name
    if model is None:
        model = exp.split("-")[0]
    # default config name follows the model name
    if config_name is None:
        config_name = f"{model}.yaml"

    if local:
        args = "--config configs/local.yaml " + args

    # subtokenizer (may not need)
    if subtokenizer is not None:
        args += f" --model.pretrained_tokenizer {Macros.work_dir}/subtokenizer/{subtokenizer}"

    # preparing output dir
    if output_dir is None:
        output_dir = Macros.work_dir / "exp" / setup / eval_set / exp / eval_method

    if output_dir.exists():
        if overwrite:
            logger.info(f"Overwriting {output_dir}")
        else:
            raise RuntimeError(f"{output_dir} already exists")
    su.io.mkdir(output_dir, fresh=True)

    # run command
    print(f"===== Eval: {setup}/{eval_set}/{exp}/{eval_method} =====")
    s = f"python -m teco.model.{model} predict"
    if (Path.cwd() / "configs" / config_name).exists():
        s += f" --config configs/{config_name}"
    s += f" --data.predict_data_dir {Macros.work_dir}/setup/{setup}/{eval_set}"
    # s += f" --data.train_data_dir {Macros.work_dir}/setup/{setup}/train/train"
    s += f" --output_dir {output_dir}"
    s += f" --no_ckpt_ok {no_ckpt_ok}"
    if len(args) > 0:
        s += f" {args}"
    print(f"+ {s}")
    c.run(s)


@task
def train_codet5(
    c,
    setup: str = "CSNm",
    suffix: str = "dev",
    overwrite: bool = False,
    use_base: Optional[str] = None,
    local: bool = False,
    debug: bool = False,
    args: str = "",
):
    if debug:
        setup += "-Debug"
    exp_train(
        c,
        setup=setup,
        exp=f"CodeT5-{suffix}",
        subtokenizer="codet5",
        overwrite=overwrite,
        use_base=use_base,
        local=local,
        args=args,
    )


@task
def nofinetune_codet5(
    c,
    setup: str = "CSNm",
    suffix: str = "noft",
    overwrite: bool = False,
    local: bool = False,
    debug: bool = False,
    args: str = "",
):
    if debug:
        setup += "-Debug"
    args += " --trainer.max_epochs=1"
    exp_train(
        c,
        setup=setup,
        exp=f"CodeT5-{suffix}",
        model="CodeT5NoFinetune",
        config_name="CodeT5.yaml",
        subtokenizer="codet5-vanilla",
        overwrite=overwrite,
        local=local,
        args=args,
    )


@task
def train_codegpt(
    c,
    setup: str = "CSNm",
    suffix: str = "dev",
    overwrite: bool = False,
    use_base: Optional[str] = None,
    local: bool = False,
    debug: bool = False,
    args: str = "",
):
    if debug:
        setup += "-Debug"
    exp_train(
        c,
        setup=setup,
        exp=f"CodeGPT-{suffix}",
        subtokenizer="codegpt",
        overwrite=overwrite,
        use_base=use_base,
        local=local,
        args=args,
    )


@task
def eval_single(
    c,
    setup: str = "CSNm",
    suffix: str = "dev",
    overwrite: bool = False,
    local: bool = False,
    debug: bool = False,
    trained: str = "CodeT5-base",
    eval_set: str = "eval-any-stmt/val",
    decoding: str = "bs10",
    args: str = "",
    use_debug_trained: bool = False,
    ckpt: str = "last",
):
    # by default, load models trained on full dataset even for debug
    setupt = setup

    if debug:
        setup += "-Debug"
        if use_debug_trained:
            setupt += "-Debug"

    model_cls = trained.split("-")[0]

    args += f" --model.model_cls={model_cls} --model.model_ckpt={Macros.work_dir}/exp/{setupt}/train/{trained}/model/{ckpt}.ckpt"

    decode_method, decode_params = DECODING_VARIANTS[decoding]
    str_decode_params = '"{' + ", ".join(f"{k}: {v}" for k, v in decode_params.items()) + '}"'
    args += f" --model.decode_method={decode_method} --model.decode_params={str_decode_params}"

    exp_eval(
        c,
        setup=setup,
        exp=f"SingleEvaluator-{suffix}",
        eval_set=eval_set,
        eval_method=f"{decoding}-{ckpt}",
        config_name="eval.yaml",
        overwrite=overwrite,
        local=local,
        args=args,
    )


@task
def gen_subset_preds(
    c,
    model: str,
    from_set: str,
    to_set: str,
    setup: str = "CSNm",
    decoding: str = "bs10-last",
):
    c.run(
        f"""
        python -m teco.eval.gen_subset_preds\
            --setup {setup}\
            --from_set {from_set}\
            --to_set {to_set}\
            run_model\
            --model {model}\
            --decoding {decoding}
        """
    )


@task
def compute_runtime_metrics(
    c,
    models: str,
    setup: str = "CSNm",
    eval_set: str = "eval-runnable-any-stmt/test",
    decoding: str = "bs10-last",
    mode: str = "top1",
    timeout_per_test: int = 5,
    batch_size: int = 1_000,
    pool_size: int = 16,
    compile_only: bool = False,
    no_compile: bool = False,
    debug: bool = False,
):
    models = [m.strip() for m in models.split(",")]
    pred_dir_list = [Macros.work_dir / "exp" / setup / eval_set / m / decoding for m in models]
    for pred_dir in pred_dir_list:
        if not pred_dir.exists():
            raise ValueError(f"{pred_dir} does not exist")
    pred_dir_list = [str(p) for p in pred_dir_list]

    c.run(
        f"""
        python -m teco.eval.compute_runtime_metrics\
            --mode {mode}\
            --timeout_per_test {timeout_per_test}\
            --debug {debug}\
            compute_metrics_batch\
            --pred_dir_list '{pred_dir_list}'\
            --data_dir {Macros.work_dir}/setup/{setup}/{eval_set}\
            --batch_size {batch_size}\
            --pool_size {pool_size}\
            --compile_only {compile_only}\
            --no_compile {no_compile}"""
    )


@task
def rerank_runnable(c, src: str, tgt: str, setup: str = "CSNm", decoding: str = "bs10-last"):
    c.run(
        f"""
        python -m teco.eval.rerank_runnable rerank\
            --src_model {src}\
            --tgt_model {tgt}\
            --setup {setup}\
            --decoding {decoding}"""
    )


@task
def pull_model(c, suffix: str = "teco-norr", repo_id: str = "EngineeringSoftware/teco"):
    c.run(
        f"""
        python -m teco.model.hf_adapter pull\
            --model_cls CodeT5\
            --exp_dir {Macros.work_dir}/exp/CSNm/train/CodeT5-{suffix}\
            --repo_id {repo_id}\
            --args "--model.inputs=[fields_notset,last_called_method,types_local,types_absent,similar_stmt_1_0,setup_teardown,focalm,sign,prev_stmts] --model.output=stmt --seed_everything=5295" """
    )
