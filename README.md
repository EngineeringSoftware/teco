# TeCo

TeCo is a deep learning model using code semantics to automatically complete the next statement in a test method. Completing tests requires reasoning about the execution of the code under test, which is hard to do with only syntax-level data that existing code completion models use. To solve this problem, we leverage the fact that tests are readily executable. TeCo extracts and uses execution-guided code semantics as inputs for the ML model, and performs reranking via test execution to improve the outputs. On a large dataset with 131K tests from 1270 open-source Java projects, TeCo outperforms the state-of-the-art by 29% in terms of test completion accuracy.

TeCo is presented in the following ICSE 2023 paper:

Title: [Learning Deep Semantics for Test Completion](https://arxiv.org/pdf/2302.10166.pdf)

Authors: Pengyu Nie, Rahul Banerjee, Junyi Jessy Li, Raymond Mooney, Milos Gligoric

## Table of Contents

- [Test Completion Corpus](#test-completion-corpus)
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Test Completion Corpus

Our test completion corpus contains several artifacts that are useful at different stages of data collection and model training/evaluation. Some large artifacts need to be downloaded separately from UTBox, and for those we provide download links and instructions to unpack the downloaded files to this repository.

### repositories list
- The list of repositories used in our study.
- In this repository: `_work/repos/filtered/`
- Contents:
  - `repos.json` is the list of repositories with metadata.
  - `licenses.tgz` contains the licenses of these repositories.

### repositories files
- The files of the repositories used in our study, archived in case some repositories are removed or renamed. To reproduce the full data collection + model training/evaluation workflow, you need to download all repositories (*large size* - 41GB). To reproduce only the model training/evaluation part, you need to download the repositories in the evaluation set (2GB).
- Download link for all repositories: https://utexas.box.com/s/n2gjzd4toy5j4t0sv4ztnngezxrx3rhc
  - Unzip the files (multi-volume zip files) with the following commands: `zip -s 0 downloads.zip --out downloads-single.zip && unzip downloads-single.zip && rm downloads-single.zip`, then move the extracted `downloads` folder to this repository at `_work/downloads`.
- Download link for repositories in the evaluation set: https://utexas.box.com/s/edmidy4h1plpmoeg5ew1bru3qg6zp8hi
  - Unzip the downloaded file with `tar xzf downloads.tgz`, then move the extracted `downloads` folder to this repository at `_work/downloads`.
- Contents:
  - each repository is stored in a separate folder, with the folder name being the `{user_name}_{repo_name}` (e.g., `apache_fluo` corresponds to github.com/apache/fluo).

### processed corpus
- The processed test completion corpus, ready for use by model training/evaluation (3GB).
- Download link: https://utexas.box.com/s/os5y2iuozindk9z9dcvjahosv427qcnw
  - Unzip the downloaded file with `tar xzf data.tgz`, then move the extracted `data` folder to this repository at `_work/data`.
- Contents:
  - the corpus is stored as multiple `jsonl` files; each file contains one data field for all examples, with each line for one example (e.g., `focalm.jsonl` contains the methods under test).
  - `config.json` contains the configuration for data filtering.
  - `filter_counter.json` contains the number of repositories/tests/statements that are filtered out.

## Pre-requisites

The following software/hardware are required to run TeCo. GPU is required for training and evaluating the model, but not required if you only use the data collection and processing scripts.

- Linux operating system
- [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) or Anaconda
- JDK 8 and Maven 3.8 (recommended to install via [sdkman](https://sdkman.io/))
- Nvidia GPU with at least 8GB memory, and appropriate driver installed

Run `./check-prereq.sh` to check if you have these pre-requisites ready.

Debugging tips: you should have these commands available in PATH: `conda`, `java`, `mvn`, `nvidia-smi` (if you want to use GPU), `nvcc` (if you want to use system-installed cuda instead of letting conda install it).

## Installation

Ensure that you met the [pre-requisites](#pre-requisites) before proceeding.

Then, you can install a conda environment for TeCo by running the following script, which includes GPU support if GPU is available:
```
./prepare-env.sh
```

After this script finishes, you can activate the conda environment by running:
```
conda activate teco
```
If this step is successful you should see a `(teco)` prefix in the command line prompt. You may need to reactivate this conda environment every time you open a new terminal.

If you need to rerun the installation script, make sure the existing conda environment is deactivated by `conda deactivate`.

You can run the following commands to quickly check if the installation is successful:
```
# try if data collection is working
inv data.collect-raw-data --debug

# try if model training is working (requires first downloading the processed corpus)
inv data.eval-setup --debug
inv exp.train-codet5 --setup CSNm-Debug --overwrite --suffix teco-norr --args "--model.inputs=[fields_notset,last_called_method,types_local,types_absent,similar_stmt_1_0,setup_teardown,focalm,sign,prev_stmts] --model.output=stmt --seed_everything=4197"
```

### Notes on GPU support and alternative CUDA installation methods

TL;DR:
- If you have an older GPU (e.g., GTX 1080 Ti) and encounter CUDA-related errors, try `./prepare-env.sh 10.2`.
- If you want to use the system-wide installed CUDA (must be 10.2/11.3/11.6) together with cuDNN and NCCL, do `./prepare-env.sh system`.

TeCo uses PyTorch 1.12.1, which requires CUDA with version 10.2/11.3/11.6, together with cuDNN and NCCL libraries. Our installation script detects whether GPU is available by checking the output of `nvidia-smi`. If GPU is not available, this script will install PyTorch in CPU-only mode, which is usually not suitable for training and evaluating the ML models (unless you know what you're doing), but enables the data collection and processing part of the TeCo to run. If GPU is available, this script will install CUDA 11.6, cuDNN, and NCCL in the conda environment. The installed CUDA is only usable when the conda environment is activated.

You can change the CUDA version installed by adding an option to the installation script: `./prepare-env.sh cuda_version`, where cuda_version can be cpu, system, 10.2, 11.3, 11.6. Use "cpu" if you want to install PyTorch in CPU-only mode even if GPU is available. Use "system" if you have already performed a system-wide installation of CUDA (must be one of 10.2/11.3/11.6), together with cuDNN and NCCL, and would like to use it instead of installing another CUDA. The default option "11.6" is usually fine especially if you're using a recent GPU, but if you're using an older GPU (e.g., GTX 1080 Ti) and encounter CUDA-related errors, you may want to try "10.2" instead.

## Usage

This section includes the entire workflow of TeCo's experiments, from data collection and processing to model training and evaluation. Unless otherwise specified, all commands should be run in the project directory and with the teco conda environment activated.

If you're only interested in the model training/evaluation part, download the part of our test completion corpus for that (the processed corpus + the repositories in the evaluation set), then jump to [Model training and evaluation](#model-training-and-evaluation).

### Data collection and processing

These steps are for collecting the data for test completion, as well as extracting the code semantics. To simplify debugging, you can add `--debug` to each command to run them on a small subset of the corpus (but debugging each step requires running the previous step on the full corpus).

*Inputs*: nothing in addition to the current repo, but in case some subject repositories have been removed from GitHub, you may download our archived repositories files (see [Corpus > repositories files](#repositories-files)).

*Outputs*: processed corpus at `_work/data`.

*Workflow*:

- [Collecting code elements](#collecting-code-elements)
- [Extracting syntax-level data](#extracting-syntax-level-data)
- [Extracting code semantics](#extracting-code-semantics)

#### Collecting code elements

This step extracts the raw code elements sets from the repositories, i.e., the collection phase in our paper. (time: ~6h)
```
inv data.collect-raw-data
```
The extracted code elements will be stored at `_work/raw-data`.

#### Extracting syntax-level data

This step processes the raw code elements sets to extract the syntax-level data (i.e., method under test, test method signature, test method statements). It also performs data filtering based on the criteria described in our paper's Section V. (time: ~1.5h)
```
inv data.process
```
The process corpus will be stored at `_work/data`.

#### Extracting code semantics

This step extracts the code semantics from the raw code elements sets (i.e., local var types, absent types, unset fields, setup teardown, last called method, similar statement). (time: ~20h)
```
inv data.extend-data-all
```
The additional code semantics data will be amended to `_work/data`.


### Model training and evaluation

These steps are for training and evaluating TeCo model after data is collected.

*Inputs*: 

- the processed corpus at `_work/data` (prepared from [Data collection and processing](#data-collection-and-processing) or downloaded from [Corpus > processed corpus](#processed-corpus)).
- the subject repositories needed during computing runtime metrics will be automatically cloned from GitHub, but in case some subject repositories have been removed from GitHub, you may download our archived repositories files (see [Corpus > repositories files](#repositories-files)).

*Outputs*:

- trained model at `_work/exp/CSNm/train/CodeT5-teco-norr`.
- evaluation results at `_work/exp/CSNm/eval-{any,runnable-any,assert,runnable-assert}-stmt/test/SingleEvaluator-teco/bs10-last`

*Workflow*:

- [Splitting corpus](#splitting-corpus)
- [Training model](#training-model) OR [Downloading model](#downloading-model)
- [Evaluating model](#evaluating-model)
- [Computing compilable/runnable status](#computing-compilablerunnable-status)
- [Reranking via execution](#reranking-via-execution)

#### Splitting corpus

```
inv data.eval-setup
```

As training and running ML models can be time-consuming, we provide a small version of the corpus after splitting for debugging purposes, CSNm-Debug, with 800/25/25 tests in train/val/eval sets. To obtain this debug version, run:
```
inv data.eval-setup --debug
```
The generated sets will be stored in `_work/setup/CSNm-Debug`. To use it in the following commands of training and evaluating the model, replace `--setup CSNm` with `--setup CSNm-Debug`.

#### Training model

The training hyper-parameters are specified in `configs/CodeT5.yaml`. The default hyper-parameters are suitable for training the model on our machine with 4 GTX 1080Ti GPUs. You may want to change `batch_size` and `gradient_accumulation_steps` to fit your GPU configurations.

To train the model: (time: ~20h on our machine with 4 GTX 1080Ti GPUs)
```
inv exp.train-codet5 --setup CSNm --overwrite --suffix teco-norr --args "--model.inputs=[fields_notset,last_called_method,types_local,types_absent,similar_stmt_1_0,setup_teardown,focalm,sign,prev_stmts] --model.output=stmt --seed_everything=4197"
```
The trained model will be stored in `_work/exp/CSNm/train/CodeT5-teco-norr`. Note that the model at this point is the variant of TeCo without reranking (i.e., TeCo-noRr in the paper). You still need to complete a few more steps below to obtain the full TeCo model.

#### Downloading model

Alternative to training the model yourself, you can download the version of the model we trained from 
[🤗 hub](https://huggingface.co/EngineeringSoftware/teco):
```
inv exp.pull-model
```
The pulled model will also be stored in `_work/exp/CSNm/train/CodeT5-teco-norr`.

#### Evaluating model

To evaluate the trained model on the eval set, run:
```
inv exp.eval-single --setup CSNm --overwrite --suffix teco-norr --trained CodeT5-teco-norr --decoding bs10 --eval-set eval-any-stmt/test --args "--seed_everything=4197"
```
The results will be stored in `_work/exp/CSNm/eval-any-stmt/test/SingleEvaluator-teco-norr/bs10-last`. The `metrics_summary.json` contains the automated metrics, and the `preds.jsonl` contains the raw predictions.

Then, compute the metrics on the runnable subset and oracle subset:
```
inv exp.gen-subset-preds --setup CSNm --model SingleEvaluator-teco-norr --from-set eval-any-stmt/test --to-set eval-runnable-any-stmt/test
inv exp.gen-subset-preds --setup CSNm --model SingleEvaluator-teco-norr --from-set eval-any-stmt/test --to-set eval-assert-stmt/test
```
The results will be stored in `_work/exp/CSNm/eval-runnable-any-stmt/test/SingleEvaluator-teco-norr/bs10-last` and `_work/exp/CSNm/eval-assert-stmt/test/SingleEvaluator-teco-norr/bs10-last`, correspondingly.

#### Computing compilable/runnable status

The next step is to check if the predicted statements are compilable/runnable on the runnable subset. First, we need to compile the repositories that will be used in the evaluation:
```
SEUTIL_SHOW_FULL_OUTPUT=1 inv exp.compute-runtime-metrics --setup CSNm --compile-only --models SingleEvaluator-teco-norr
```
If any compilation error occurs, it means the runtime environment (e.g., JDK 8 and Maven) is not properly set up. Please check the error messages and fix them. The compilation of repositories only need to be done once and can be used for evaluating the compilable/runnable status of any number of models.

Then, compute the compilable/runnable status of the TeCo-noRr model: (time: ~12h)
```
inv exp.compute-runtime-metrics --setup CSNm --no-compile --models SingleEvaluator-teco-norr --mode="first-runnable" --pool-size=32 --batch-size=200
```
The results will be amended to `_work/exp/CSNm/eval-runnable-any-stmt/test/SingleEvaluator-teco-norr/bs10-last`.

Finally, we can compute the metrics (with %compile/%run) on the runnable-oracle subset:
```
inv exp.gen-subset-preds --setup CSNm --model SingleEvaluator-teco-norr --from-set eval-runnable-any-stmt/test --to-set eval-runnable-assert-stmt/test
```

#### Reranking via execution

The last step is to perform reranking based on the compilable/runnable status:
```
inv exp.rerank-runnable --setup CSNm --src SingleEvaluator-teco-norr --tgt SingleEvaluator-teco
```
This will generate the results of the full TeCo model on the evaluation set and its three subsets, located at the following places.
eval set: `_work/exp/CSNm/eval-any-stmt/test/SingleEvaluator-teco/bs10-last`
runnable subset: `_work/exp/CSNm/eval-runnable-any-stmt/test/SingleEvaluator-teco/bs10-last`
oracle subset: `_work/exp/CSNm/eval-assert-stmt/test/SingleEvaluator-teco/bs10-last`
runnable-oracle subset: `_work/exp/CSNm/eval-runnable-assert-stmt/test/SingleEvaluator-teco/bs10-last`


## Citation

If you have used Teco in a research project, please cite the research paper in any related publication:
```
@inproceedings{NieETAL22Teco,
  title =        {Learning Deep Semantics for Test Completion},
  author =       {Pengyu Nie and Rahul Banerjee and Junyi Jessy Li and Raymond J. Mooney and Milos Gligoric},
  booktitle =    {International Conference on Software Engineering},
  year =         {2023},
}
```
