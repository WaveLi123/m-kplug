"""
https://github.com/pytorch/fairseq/pull/2172/files#diff-dadc6e34f003c219981146fc58b24945

z-loss是什么玩意？
"""


import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('cross_entropy')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, z_loss=None):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.z_loss = z_loss

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--z-loss',
            default=0.0,
            type=float,
            help='Add loss equal to z_loss * log(z)^2, where z are scores before softmax.')
    """
    z-loss是什么玩意？
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        if self.z_loss and model.training == True:
            loss += self.compute_z_loss(net_output[0], -1)
        return loss, loss

    def compute_z_loss(self, x, vocab_dim):
        """Compute z-loss. 很关键。。。
        Add self.z_loss * log(z)^2, where x are scores before softmax. The goal of z-loss is to keep
        logits from drifting too far from zero. Originally taken from Tensorflow Mesh.
        """
        z_log = torch.logsumexp(x, vocab_dim)
        out = self.z_loss * torch.sum(torch.pow(z_log, 2))
        return out

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True