import random
import sys

import seutil as su
from invoke import task
from teco.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)


@task
def search_repos(c, debug: bool = False):
    if debug:
        lists = [Macros.work_dir / "repos" / "csn-lists" / "debug.txt"]
        out_dir = Macros.work_dir / "repos" / "csn-debug"
    else:
        lists = [
            Macros.work_dir / "repos" / "csn-lists" / f"{sn}.txt"
            for sn in ["train", "val", "test"]
        ]
        out_dir = Macros.work_dir / "repos" / "csn"
    lists = [str(p) for p in lists]

    su.io.mkdir(out_dir, fresh=True)
    c.run(
        f"""python -m teco.data.collect_repos search_repos_from_lists\
            --lists [{','.join(lists)}]\
            --out_dir {out_dir} """
    )


@task
def filter_repos(c, debug: bool = False):
    repos_file = Macros.work_dir / "repos" / "csn" / "repos.json"
    if debug:
        su.io.dump(
            Macros.debug_dir / "repos-debug.json",
            su.io.load(repos_file)[:10],
            su.io.Fmt.jsonPretty,
        )
        repos_file = Macros.debug_dir / "repos-debug.json"

        out_dir = Macros.work_dir / "repos" / "filtered-debug"
    else:
        out_dir = Macros.work_dir / "repos" / "filtered"

    su.io.mkdir(out_dir, fresh=True)

    c.run(
        f"""python -m teco.data.collect_repos filter_licensed_maven_compilable\
            --repos_file {repos_file}\
            --out_dir {out_dir} """
    )


@task
def collect_raw_data(
    c, skip_collected: bool = True, debug: bool = False, seed: int = None
):
    if debug and seed is None:
        seed = random.randint(0, 2**32)
        print(f"For reproducing the debug samples: {seed=}")

    repos_file = Macros.work_dir / "repos" / "filtered" / "repos.json"
    if debug:
        repos = su.io.load(repos_file)
        repos = repos[:3]
        su.io.dump(
            Macros.debug_dir / "repos-debug.json",
            repos,
            su.io.Fmt.jsonPretty,
        )
        repos_file = Macros.debug_dir / "repos-debug.json"
        out_dir = Macros.work_dir / "raw-data-debug"
    else:
        out_dir = Macros.work_dir / "raw-data"

    su.io.mkdir(out_dir)

    c.run(
        f"""python -m teco.data.collect_raw_data\
            --debug {debug}\
            collect_raw_data\
            --repos_file {repos_file}\
            --out_dir {out_dir}\
            --skip_collected {skip_collected} """
    )

    if debug:
        print(f"gathering sample data to {Macros.debug_dir}/raw-data-samples")
        sys.setrecursionlimit(10000)
        for p in [x["full_name"] for x in su.io.load(repos_file)]:
            for n in ["joint.class.json", "joint.method.json", "joint.field.json"]:
                if (out_dir / p / n).exists():
                    data = su.io.load(out_dir / p / n)
                    random.seed(seed)

                    if n == "joint.method.json":
                        data1 = [d for d in data if d["is_test"] and "code" in d]
                        data2 = [d for d in data if d["is_test"] and "code" not in d]
                        data3 = [
                            d for d in data if not d["is_test"] and "code" not in d
                        ]
                        random.shuffle(data1)
                        random.shuffle(data2)
                        random.shuffle(data3)
                        data = data1[:100] + data2[:10] + data3[:10]
                    else:
                        random.shuffle(data)
                        data = data[:10]

                    su.io.dump(
                        Macros.debug_dir / "raw-data-samples" / p / n,
                        data,
                        su.io.Fmt.jsonNoSort,
                    )


@task
def process(c, debug: bool = False):
    repos_file = Macros.work_dir / "repos" / "filtered" / "repos.json"
    if debug:
        repos = su.io.load(repos_file)
        repos = repos[:3]
        su.io.dump(
            Macros.debug_dir / "repos-debug.json",
            repos,
            su.io.Fmt.jsonPretty,
        )
        repos_file = Macros.debug_dir / "repos-debug.json"
        out_dir = Macros.work_dir / "data-debug"
    else:
        out_dir = Macros.work_dir / "data"

    su.io.mkdir(out_dir, fresh=True)

    s = f"""python -m teco.data.process_raw_data\
        --seed 42\
        process\
        --repos_file {repos_file}\
        --raw_data_dir {Macros.work_dir}/raw-data\
        --out_dir {out_dir} """
    c.run(s)


@task
def extend_data(c, name: str, debug: bool = False, args: str = ""):
    data_dir = Macros.work_dir / "data"
    if debug:
        data_dir = Macros.work_dir / "data-debug"

    if name.startswith("runtime"):
        c.run(
            f"""python -m teco.data.extend_data_runtime\
                collect\
                --name {name}\
                --data_dir {data_dir}\
                --repos_file {Macros.work_dir}/repos/filtered/repos.json\
                --downloads_dir {Macros.downloads_dir} {args}"""
        )
    else:
        c.run(
            f"""python -m teco.data.extend_data\
                --data_dir {data_dir}\
                --raw_data_dir {Macros.work_dir}/raw-data\
                collect_single --name {name} {args}"""
        )


@task
def extend_data_all(c, debug: bool = False):
    data_dir = Macros.work_dir / "data"
    if debug:
        data_dir = Macros.work_dir / "data-debug"

    # collect code semantics from static analysis
    for name in [
        "types_local",
        "types_absent",
        "setup_teardown_methods",
        "fields_set_notset",
        "last_called_methods",
        "similar_stmts",
    ]:
        print(f">>> Collecting {name}...")
        c.run(
            f"""python -m teco.data.extend_data\
                --data_dir {data_dir}\
                --raw_data_dir {Macros.work_dir}/raw-data\
                collect_single --name {name}"""
        )

    # collect runtime data to decide if original test is runnable (only need to be done on evaluation set)
    print(f">>> Collecting runtime_types_values...")
    c.run(
        f"""python -m teco.data.extend_data_runtime\
            collect\
            --name runtime_types_values\
            --data_dir {data_dir}\
            --repos_file {Macros.work_dir}/repos/filtered/repos.json\
            --downloads_dir {Macros.downloads_dir}\
            --only_sets='[val,test]' --overwrite=True"""
    )


@task
def eval_setup(c, mask_str_literal: bool = True, debug: bool = False):
    setup_name = "CSN"
    if mask_str_literal:
        setup_name += "m"
        setup_arg = "--mask_str_literal True"
    else:
        setup_arg = ""

    if debug:
        setup_name += "-Debug"
        split_arg = "--limit_train 800 --limit_val 25 --limit_test 25 --seed 7"
    else:
        split_arg = ""

    setup_dir = Macros.work_dir / "setup" / setup_name

    repos_file = Macros.work_dir / "repos" / "filtered" / "repos.json"
    data_dir = Macros.work_dir / "data"
    split_dir = setup_dir / "split"

    su.io.mkdir(setup_dir, fresh=True)

    # get split
    c.run(
        f"""python -m teco.data.get_split\
            get_split_from_repos\
            --repos_file {repos_file}\
            --data_dir {data_dir}\
            --out_dir {split_dir}\
            {split_arg} """
    )

    # prepare train
    c.run(
        f"""python -m teco.data.eval_setup\
            {setup_arg}\
            prepare_train\
            --data_dir {data_dir}\
            --split_dir {split_dir}\
            --out_dir {setup_dir}/train """
    )

    # prepare eval
    c.run(
        f"""python -m teco.data.eval_setup\
            {setup_arg}\
            prepare_eval\
            --data_dir {data_dir}\
            --split_dir {split_dir}\
            --out_dir {setup_dir}/eval-any-stmt\
            --seed 42 """
    )
    c.run(
        f"""python -m teco.data.eval_setup\
            {setup_arg}\
            prepare_eval\
            --data_dir {data_dir}\
            --split_dir {split_dir}\
            --out_dir {setup_dir}/eval-assert-stmt\
            --first_assertion True --seed 42 """
    )
    c.run(
        f"""python -m teco.data.eval_setup\
            {setup_arg}\
            prepare_eval\
            --data_dir {data_dir}\
            --split_dir {split_dir}\
            --out_dir {setup_dir}/eval-runnable-any-stmt\
            --runnable True --seed 42 """
    )
    c.run(
        f"""python -m teco.data.eval_setup\
            {setup_arg}\
            prepare_eval\
            --data_dir {data_dir}\
            --split_dir {split_dir}\
            --out_dir {setup_dir}/eval-runnable-assert-stmt\
            --runnable True --first_assertion True --seed 42 """
    )


@task
def gen_atlas_dataset(c, setup: str = "CSNm"):
    c.run(
        f"""
        python -m teco.data.gen_atlas_dataset\
            generate\
            --setup {setup}\
            --out_dir {Macros.work_dir}/setup/{setup}-atlas
        """
    )
    c.run(f"mkdir {Macros.work_dir}/setup/{setup}-atlas/Vocabulary")


@task
def gen_toga_dataset(c, setup: str = "CSNm"):
    c.run(
        f"""
        python -m teco.data.gen_toga_dataset\
            generate\
            --setup {setup}\
            --out_dir {Macros.work_dir}/setup/{setup}-toga
        """
    )
