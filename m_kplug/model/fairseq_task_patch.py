import os
from fairseq.tasks import FairseqTask
from fairseq.tasks.translation import TranslationTask, logger, data_utils, utils
from fairseq.data.dictionary import Dictionary
from .bert_dictionary import BertDictionary


@classmethod
def load_dictionary(cls, filename, bertdict=False):
    if bertdict:
        return BertDictionary.load_from_file(filename)
    return Dictionary.load(filename)

FairseqTask.load_dictionary = load_dictionary


_add_args = TranslationTask.add_args


@staticmethod
def add_args(parser):
    _add_args(parser)
    parser.add_argument('--bertdict', action='store_true', default=False,
                        help='use bert dictionary')

@classmethod
def setup_task(cls, args, **kwargs):
    """Setup the task (e.g., load dictionaries).
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    args.left_pad_source = utils.eval_bool(args.left_pad_source)
    args.left_pad_target = utils.eval_bool(args.left_pad_target)

    paths = utils.split_paths(args.data)
    assert len(paths) > 0
    # find language pair automatically
    if args.source_lang is None or args.target_lang is None:
        args.source_lang, args.target_lang = data_utils.infer_language_pair(
            paths[0]
        )
    if args.source_lang is None or args.target_lang is None:
        raise Exception(
            "Could not infer language pair, please provide it explicitly"
        )

    # load dictionaries
    src_dict = cls.load_dictionary(
        os.path.join(paths[0], "dict.{}.txt".format(args.source_lang)),
        bertdict=args.bertdict  # 改动1
    )
    tgt_dict = cls.load_dictionary(
        os.path.join(paths[0], "dict.{}.txt".format(args.target_lang)),
        bertdict=args.bertdict  # 改动2
    )
    assert src_dict.pad() == tgt_dict.pad()
    assert src_dict.eos() == tgt_dict.eos()
    assert src_dict.unk() == tgt_dict.unk()
    logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
    logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))

    return cls(args, src_dict, tgt_dict)


TranslationTask.add_args = add_args
TranslationTask.setup_task = setup_task
