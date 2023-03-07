from typing import List

import seutil as su

logger = su.log.get_logger(__name__)


class GenerationMixin:
    """
    Unified generation interface of models that work by consuming some
    tokens and outputing some tokens.

    This class defines a "generate" method as the main entry point for
    model generation, which:
    * does some common preprocessing of inputs in "gen_preprocess"
      (e.g., tokenize, run encoder);
    * dispatches to specific generation methods (generate_XXX);
    * does some common postprocessing of each output in the list in
      "gen_postprocess" (e.g., detokenize);
    * returns a list of sorted top-k generation outputs; sorting is
      done in "sort_topk".

    For flexibility, this interface does not constrain the number or
    names of input arguments (except for "decode_method" which is
    reserved for dispatching to different generation methods), nor the
    number or names of fields in the returned list of dicts. However,
    the output should be serializable (e.g., contains only primitive
    types + list + dict).  The following input args / output fields
    are recommanded:

    Input args:
        seq: InputSequence, the input sequence (tokens + subtoken ids)
          to the model

    Output fields:
        toks: List[str], sequence of tokens outputed by the model

        score: float, the likelihood (usually log probability) of the
          generated sequence; it is used by the default sort_topk
          implementation, which can be overridden if necessary

        weight: float, the importance measurement of the generated
          sequence (e.g., the number of times being generated in a
          sampling algorithm)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_warned_about_kwargs = False

    def generate(self, decode_method: str, **decode_params) -> List[dict]:
        """
        Dispatches to specific generation method by name.
        The decode_method is usually "greedy", "sampling", or "beam_search".
        """
        gen_method = getattr(self, f"generate_{decode_method}")
        topk = gen_method(**self.gen_preprocess(decode_params))
        for gen_out in topk:
            self.gen_postprocess(gen_out)
        sort_topk(topk)
        return topk

    def gen_preprocess(self, decode_params: dict) -> dict:
        """
        Common preprocessing for generation; default implementation
        does nothing.
        """
        return decode_params

    def gen_postprocess(self, gen_out: dict) -> None:
        """
        Common postprocessing for each generation output; default
        implementation does nothing.

        To avoid copying, the gen_out dict is modified in-place.
        """
        pass

    def sort_topk(self, topk: List[dict]):
        """
        Sorts the topk generation outputs by score.

        To avoid copying, the topk list is modified in-place.

        Default implementaion expect the "score" field.
        """
        topk.sort(key=lambda x: x["score"], reverse=True)

    def generation_warn_unused_kwargs(self, kwargs):
        """
        Can be used at beginning of "generate_X" methods to warn about
        unused input arguments (due to typo or misconfiguration).
        """
        if not self.generation_warned_about_kwargs and len(kwargs) > 0:
            logger.warning(f"Some decode parameters are not used: {kwargs}")
            self.generation_warned_about_kwargs = True


def sort_topk(topk: List[dict]):
    """
    Sorts the topk generation outputs by score.
    Default implementaion expect "score" key in each generation output.
    """
    topk.sort(key=lambda x: x["score"], reverse=True)
