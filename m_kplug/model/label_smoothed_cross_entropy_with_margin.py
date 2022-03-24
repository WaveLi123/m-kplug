# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def label_smoothed_nll_loss_margin(lprobs, target, epsilon, margin_loss, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    margin_loss = margin_loss.repeat(nll_loss.size(0))
    margin_loss = margin_loss.unsqueeze(-1)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        margin_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        margin_loss = margin_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        margin_loss = margin_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    loss += margin_loss
    return loss, nll_loss, margin_loss, (1.0 - epsilon) * nll_loss + eps_i * smooth_loss


@register_criterion("label_smoothed_cross_entropy_with_margin", dataclass=LabelSmoothedCrossEntropyCriterionConfig)
class LabelSmoothedCrossEntropyCriterionMargin(LabelSmoothedCrossEntropyCriterion):
    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if 'margin_loss' in net_output[-1]:
            margin_loss = net_output[-1]['margin_loss']
            loss, nll_loss, new_margin_loss, origin_loss = label_smoothed_nll_loss_margin(
                lprobs,
                target,
                self.eps,
                margin_loss=margin_loss,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            # sys.stdout.write("[loss]: {}, [nll_loss]: {}, [margin_loss]: {}, [origin_loss]: {}, [single_margin_loss]: {}\n".format(loss, nll_loss, new_margin_loss, origin_loss, margin_loss))
        else:
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )

        return loss, nll_loss
