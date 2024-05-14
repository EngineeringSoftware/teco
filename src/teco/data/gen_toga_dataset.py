import re
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import javalang
import numpy as np
import seutil as su
from jsonargparse import CLI
from tqdm import tqdm

from teco.data.data import BASIC_FIELDS, Data
from teco.data.eval_setup import EvalSetup
from teco.data.utils import load_dataset
from teco.macros import Macros

whitespace_re = re.compile(r"\s+")

split_re = re.compile('"<AssertPlaceHolder>" ;[ }]*')

long_re = re.compile("(assertEquals\([-]?[0-9]+)L(, long[0-9]*\))")
cast_re = re.compile("(assertEquals\()\(\w+\)(\S*, \w+\))")
paren_re = re.compile("(assertEquals\()\((\S+)\)(, \w+\))")
equals_bool_re = re.compile("assertEquals \( (false|true) , (.*) \)")

arg_re_generic = re.compile("assert\w*\s*\((.*)\)")

vocab = None

TEMPLATES = ["assertTrue", "assertFalse", "assertEquals", "assertNotNull", "assertNull"]


def clean(code):
    return whitespace_re.sub(" ", code).strip()


def parser_type_to_java_type(t):
    try:
        t = t.value if "value" in dir(t) else t.member
    except AttributeError:
        return None

    if t == "true" or t == "false":
        return bool
    try:
        t = int(t)
        return int
    except ValueError:
        try:
            t = float(t)
            return float
        except ValueError:
            return str


def get_type(assertion_type, assertion, arg, full_test):
    full_test = full_test.replace('"<AssertPlaceHolder>" ;', assertion + " ; ")

    tokens = javalang.tokenizer.tokenize(full_test)
    parser = javalang.parser.Parser(tokens)
    try:
        test_obj = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, TypeError, IndexError, StopIteration):
        return None

    target_type = None

    if assertion_type == "assertTrue" or assertion_type == "assertFalse":
        target_type = bool

    elif assertion_type == "assertEquals":
        target_type = parser_type_to_java_type(arg)

    if (
        assertion_type == "assertNotNull"
        or assertion_type == "assertNull"
        and isinstance(arg, javalang.tree.MethodInvocation)
    ):
        return None

    all_var_types = []
    all_vars = []
    for p, node in test_obj:
        if isinstance(node, javalang.tree.LocalVariableDeclaration):
            name = node.declarators[0].name

            if "member" in dir(arg) and name == arg.member and not target_type:
                target_type = pretty_type(node.type)
            else:
                all_var_types += [pretty_type(node.type)]
                all_vars += [node.declarators[0].name]

        elif isinstance(node, javalang.tree.Literal):
            all_var_types += [parser_type_to_java_type(node.value)]
            all_vars += [node.value]

    if not target_type:
        return None

    same_type_vars = []
    for var, _type in zip(all_vars, all_var_types):
        if _type == target_type:
            same_type_vars += [var]

    return target_type, same_type_vars


def pretty_type(tree, full_type=""):
    if isinstance(tree, javalang.tree.BasicType):
        return tree.name

    if tree.sub_type:
        return tree.name + "." + pretty_type(tree.sub_type, full_type)
    return tree.name


def get_type_info(assertion, test_method):
    start = 0

    if not test_method:
        return None

    end = assertion.find("(")

    tokens = javalang.tokenizer.tokenize(assertion)
    parser = javalang.parser.Parser(tokens)

    try:
        assertion_obj = parser.parse_primary()
    except javalang.parser.JavaSyntaxError:
        return False

    try:
        if len(assertion_obj.arguments) > 2:
            # print("more than 2 args")
            return False
    except AttributeError:
        return False

    # find assertion type
    assertion_type = assertion[start:end].strip()
    if not assertion_type in TEMPLATES:
        # print("non template assertion type")
        return False

    # IF there is only 1 arg -> then use first arg
    # OTHERWISE, find the method invocation

    relevant_arg = None
    other_arg = None
    total_args = len(assertion_obj.arguments)
    arg_num = -1
    if len(assertion_obj.arguments) == 1:
        relevant_arg = assertion_obj.arguments[0]
    else:
        for arg_idx, arg in enumerate(assertion_obj.arguments):
            if isinstance(arg, javalang.tree.MethodInvocation):
                relevant_arg = arg
                arg_num = arg_idx
            else:
                other_arg = arg

    if not relevant_arg:
        # print("non typeable arg")
        return False

    if not other_arg:
        other_arg = relevant_arg

    out = get_type(assertion_type, assertion, other_arg, test_method)
    if not out:
        return False

    _type, matching_type_vars = out

    return _type, arg_num, total_args, matching_type_vars


def assertion_to_arg(assertion, arg_num, total_args):
    m = arg_re_generic.search(assertion)

    g = m.group(1)
    args = g.split(",")
    try:
        assert len(args) == total_args and total_args <= 2 and len(args) > arg_num
    except AssertionError as e:
        if total_args == 1:
            return g
        else:
            raise e

    return args[arg_num]


def gen_varients(_type, arg, matching_type_vars):

    out = []
    values = matching_type_vars
    arg = arg.strip()

    if _type in vocab:
        top_values = list(vocab[_type].keys())
        if _type == int:
            top_values = [
                int(x.replace("(", "").replace(")", "").replace(" ", ""))
                for x in top_values
            ]
        elif _type == float:
            top_values = [
                float(
                    x.replace("(", "")
                    .replace(")", "")
                    .replace(" ", "")
                    .replace("Complex.", "")
                )
                for x in top_values
            ]
        elif _type == str:
            top_values += ["'" + k + "'" for k in vocab[_type].keys()]

        values = top_values + values

    for var in values:
        out += ["assertEquals ( {} , {} )".format(var, arg)]

    if _type == bool:
        out += ["assertTrue ( {} )".format(arg), "assertFalse ( {} )".format(arg)]
    elif not _type == int and not _type == float:
        out += ["assertNotNull ( {} )".format(arg), "assertNull ( {} )".format(arg)]

    return list(set(out))


class TogaDataGenerator:
    def __init__(
        self, vocab_path: su.arg.RPath = Path(__file__).parent / "toga_vocab.npy"
    ):
        self.vocab_path = vocab_path

        # build global vocab
        vocab_src = np.load(self.vocab_path, allow_pickle=True).item()

        self.vocab_eval = {}
        for k, v in vocab_src.items():
            self.vocab_eval[k] = {
                k2: v2
                for k2, v2 in list(
                    reversed(sorted(v.items(), key=lambda item: item[1]))
                )[0:8]
            }

        self.vocab_noneval = {}
        for k, v in vocab_src.items():
            self.vocab_noneval[k] = {
                k2: v2
                for k2, v2 in list(
                    reversed(sorted(v.items(), key=lambda item: item[1]))
                )[0:5]
            }

    def format_data(
        self, data: Data, stmt_i: int, gold: Optional[List[str]] = None
    ) -> Tuple[str, str, str]:
        src_toks = (
            data.test_sign_toks
            + ["{"]
            + sum(data.test_stmt_toks[:stmt_i], [])
            + ['"<AssertPlaceHolder>"', ";"]
            + ["}"]
        )
        if gold is None:
            tgt_toks = data.test_stmt_toks[stmt_i]
        else:
            tgt_toks = gold
        return " ".join(tgt_toks), " ".join(src_toks), " ".join(data.focalm_toks)

    def to_csv(self, samples: List[tuple]) -> str:
        csv = "idx,label,fm,test,assertion\n"
        for idx, label, fm, test, assertion in samples:
            csv += f"{idx},{label},"
            csv += '"' + fm.replace('"', '""') + '",'
            csv += '"' + test.replace('"', '""') + '",'
            csv += '"' + assertion.replace('"', '""') + '"\n'
        return csv

    def generate(
        self,
        setup: str,
        out_dir: su.arg.RPath,
        val_set: str = "eval-assert-stmt/val",
        test_set: str = "eval-assert-stmt/test",
    ):
        su.io.mkdir(out_dir)
        data_dir = Macros.work_dir / "setup" / setup

        # train set
        pbar = tqdm(desc="preparing train set")
        train_dataset: List[Data] = load_dataset(
            data_dir / "train" / "train", clz=Data, only=BASIC_FIELDS, pbar=pbar
        )
        train_samples = []
        pbar.reset(len(train_dataset))
        for i, data in enumerate(train_dataset):
            # find first assertion, if any
            for stmt_i, stmt in enumerate(data.test_stmt_toks):
                if EvalSetup.is_assertion(stmt):
                    train_samples += self.generate_samples(
                        *self.format_data(data, stmt_i), index=i
                    )
                    break

            pbar.update(1)
        pbar.close()
        su.io.dump(out_dir / "train.csv", self.to_csv(train_samples), su.io.Fmt.txt)
        print(
            f"# train data: {len(train_samples)} samples (from {len(train_dataset)} methods)"
        )

        # val set
        pbar = tqdm(desc="preparing val set")
        val_dataset: List[Data] = load_dataset(
            data_dir / val_set, clz=Data, only=BASIC_FIELDS, pbar=pbar
        )
        val_golds = su.io.load(data_dir / val_set / "gold_stmts.jsonl")
        val_samples = []
        for i, (data, gold) in enumerate(zip(val_dataset, val_golds)):
            val_samples += self.generate_samples(
                *self.format_data(data, len(data.test_stmts), gold), index=i
            )
            pbar.update(1)
        pbar.close()
        su.io.dump(out_dir / "valid.csv", self.to_csv(val_samples), su.io.Fmt.txt)
        print(f"# val data: {len(val_samples)}")

        # test set
        pbar = tqdm(desc="preparing test set")
        test_dataset: List[Data] = load_dataset(
            data_dir / test_set, clz=Data, only=BASIC_FIELDS, pbar=pbar
        )
        test_golds = su.io.load(data_dir / test_set / "gold_stmts.jsonl")
        test_samples = []
        for i, (data, gold) in enumerate(zip(test_dataset, test_golds)):
            test_samples += self.generate_samples(
                *self.format_data(data, len(data.test_stmts), gold),
                index=i,
                eval_mode=True,
            )
            pbar.update(1)
        pbar.close()
        su.io.dump(out_dir / "test.csv", self.to_csv(test_samples), su.io.Fmt.txt)
        print(f"# test data: {len(test_samples)}")

    def generate_samples(
        self,
        assertion: str,
        test_prefix: str,
        focal_method: str,
        index: int,
        eval_mode: bool = False,
    ) -> List[Tuple[int, str, str, str]]:
        """Generate a list of (label, fm, test, assertion) samples out of a data (potentially empty list in eval mode)."""

        global vocab
        if eval_mode:
            vocab = self.vocab_eval
        else:
            vocab = self.vocab_noneval

        # data pre-process
        start = len("org . junit . Assert . ")
        if assertion.startswith("org . junit . Assert . "):
            assertion = assertion[start:]
        m = equals_bool_re.match(assertion)
        if m:
            if m.group(1) == "true":
                assertion = "assertTrue ( {} )".format(m.group(2))
            else:
                assertion = "assertFalse ( {} )".format(m.group(2))

        # generate candidates
        out = get_type_info(assertion, test_prefix)
        if not out:
            return []

        _type, arg_num, total_args, matching_type_vars = out
        try:
            arg_txt = assertion_to_arg(assertion, arg_num, total_args)
        except AssertionError:
            return []
        template_asserts = gen_varients(_type, arg_txt, matching_type_vars)

        if assertion not in template_asserts:
            if m := paren_re.search(assertion):
                assertion = m.group(1) + m.group(2) + m.group(3)
            if m := long_re.search(assertion):
                assertion = m.group(1) + m.group(2)
            elif m := cast_re.search(assertion):
                assertion = m.group(1) + m.group(2)

        assertion_clean = clean(assertion)

        all_samples = []
        pos_in_search_space = False
        for i in range(len(template_asserts)):
            if assertion_clean == clean(template_asserts[i]):
                pos_in_search_space = True
                all_samples.append(
                    (index, 1, focal_method, test_prefix, assertion_clean)
                )
            else:
                all_samples.append(
                    (index, 0, focal_method, test_prefix, clean(template_asserts[i]))
                )
        if not eval_mode and not pos_in_search_space:
            # make sure the positive sample is added for training
            all_samples.append((index, 1, focal_method, test_prefix, assertion_clean))

        return all_samples


if __name__ == "__main__":
    CLI(TogaDataGenerator, as_positional=False)
    # example
    # assertion = "org . junit . Assert . assertEquals ( 0 , users . size ( ) )"
    # test_prefix = 'getUsersWaitingNotificationNoWatchExpectEmptyList ( ) { net . jforum . repository . TopicWatchRepository dao = this . newDao ( ) ; net . jforum . entities . Topic topic = new net . jforum . entities . Topic ( ) ; topic . setId ( 13 ) ; java . util . List < net . jforum . entities . User > users = dao . getUsersWaitingNotification ( topic ) ; "<AssertPlaceHolder>" ; }'
    # focal_method = (
    #     'getUsersWaitingNotification ( net . jforum . entities . Topic ) { java . util . List < net . jforum . entities . User > users = session . createQuery ( ( "select<sp>u<sp>from<sp>TopicWatch<sp>tw<sp>" + ( "<sp>inner<sp>join<sp>tw.user<sp>u<sp>where<sp>tw.topic<sp>=<sp>:topic<sp>'
    #     + '<sp>and<sp>(tw.read<sp>=<sp>true<sp>or<sp>u.notifyAlways<sp>=<sp>true)" ) ) ) . setEntity ( "topic" , topic ) . setComment ( "topicWatchDAO.getUsersWaitingNotification" ) . list ( ) ; if ( ( users . size ( ) ) > 0 ) { this . markAllAsUnread ( topic ) ; } return users ; }'
    # )
    # samples = TogaDataGenerator(
    #     str(Path(__file__).parent / "vocab.npy")
    # ).generate_samples(assertion, test_prefix, focal_method, split="train")
    # for sample in samples:
    #     print(sample)
    # print(f"pos: {pos}")
    # for ne in neg:
    #     print("neg: ")
    #     print(ne)
