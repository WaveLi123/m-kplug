"""

"""

# coding: utf-8

import pdb
import torch
import torch.nn.functional as F
import math

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss, LabelSmoothedCrossEntropyCriterion


def page_rank(self_attn, src_lengths, n_step=2, dumping_factor=None):
    """
    n_step:  0,1,2,3...
    dumping_factor: [0.8, 1]
    """
    dumping_factor = 0.85 if not dumping_factor else dumping_factor
    sum_attn = torch.sum(self_attn, dim=1).unsqueeze(1) + 1e-6  # 避免出现nan
    trans_p = self_attn / sum_attn  # 转移概率矩阵，即出度归一化，会出现nan，但是后续取loss有pad，应该不影响吧？
    trans_mean = torch.sum(trans_p, dim=-1) / src_lengths.view(-1, 1)  # 不能直接用mean，因为pad会影响
    if n_step == 0:  # 近似稳态
        return trans_mean
    stable_attn = trans_mean.unsqueeze(1)
    trans_p = trans_p.transpose(1, 2)
    for i in range(n_step):
        if dumping_factor == 1:
            stable_attn = stable_attn.bmm(trans_p)
        else:
            stable_attn = dumping_factor * stable_attn.bmm(trans_p) + (1 - dumping_factor) * stable_attn
    return stable_attn.view_as(trans_mean)


# def get_copy_prior(self_attn, src_lengths, prior_type, dumping_factor=None):
def get_self_attn_guidance(self_attn, src_lengths, sag_type, dumping_factor=None):
    """ self attn guidance for copy mechanism
    或者叫 guidance_type_for_copy, copy_guidance
          get_guidance_from_self_attn_for_copy
          get_guidance_from_self_attn
    guidance_type: out_mean in_pagerank_0 in_pagerank_1 in_pagerank_2


    官方transformer的self_attn，图省事，只在一个维度上进行了mask。
    示例： 7个词，3个[PAD]，出度求和，pagerank计算，都要注意细节。
    self_attn=
    tensor([[0.0000, 0.0000, 0.0000, 0.2776, 0.3566, 0.0140, 0.3517],   # 前三行不应该影响对列求和，要mask掉
            [0.0000, 0.0000, 0.0000, 0.2776, 0.3566, 0.0140, 0.3517],   #
            [0.0000, 0.0000, 0.0000, 0.2776, 0.3566, 0.0140, 0.3517],
            [0.0000, 0.0000, 0.0000, 0.0879, 0.1638, 0.0244, 0.7239],   # 有效
            [0.0000, 0.0000, 0.0000, 0.2791, 0.1987, 0.0198, 0.5024],
            [0.0000, 0.0000, 0.0000, 0.3784, 0.4875, 0.0026, 0.1314],
            [0.0000, 0.0000, 0.0000, 0.0129, 0.0135, 0.0020, 0.9716]],
    """
    if sag_type == 'out-mean':  # 出度的均值
        return torch.sum(self_attn, dim=1) / src_lengths.view(-1, 1)
    if sag_type.startswith('in-pagerank-'):
        n_step = int(sag_type[12:])
        return page_rank(self_attn, src_lengths, n_step=n_step, dumping_factor=dumping_factor)
    raise ValueError("prior_type error, valid examples ['out-mean', 'in-pagerank-0', 'in_pagerank_1', 'in_pagerank_2']")


def get_cumulative_copy_dist(decoder_attn, p_gen=None, dist_type='mean'):
    """ 累计copy概率, cumulative distribution
    return_type: mean | max
    :deprecated
    """
    if p_gen is not None:  # scaled attn
        return torch.mean((1 - p_gen) * decoder_attn, dim=1)
    if dist_type == 'mean':
        return torch.mean(decoder_attn, dim=1)  # TODO: 1. tgt_length问题， 2. 加上softmax
    if dist_type == 'max':  # update @20200422
        return torch.max(decoder_attn, dim=1)[0]  #


@register_criterion("label_smoothed_cross_entropy_with_guidance")
class LabelSmoothedCrossEntropyCriterionWithGuidance(
    LabelSmoothedCrossEntropyCriterion
):
    def get_guidance_loss(self, net_output, sag_type='mean', dist_type='mean', dumping_factor=0.85):
        """
        Args:
            sag_type: self attention guidance type
            copy_prior:
            copy_dist:
             the sum of attention distributions over all previous decoder timesteps

        Doc:
            context free prior:  词汇的固有的重要度、一致性
            1. src_stable_self_attn:  as cumulative prior
        """

        # pdb.set_trace()
        _, extra = net_output
        src_tokens = extra['src_tokens']
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1).float()

        prior_dist = get_self_attn_guidance(extra['encoder_self_attn'], src_lengths, sag_type=sag_type,
                                            dumping_factor=dumping_factor)
        # copy_dist = get_cumulative_copy_dist(extra['attn'], extra['p_gen'])
        copy_dist = get_cumulative_copy_dist(extra['attn'], dist_type=dist_type)

        prior_dist = torch.clamp(prior_dist, 1e-9, 1 - 1e-9)
        copy_loss = F.kl_div(torch.log(prior_dist), copy_dist, reduction='none')
        non_pad_mask = extra['src_tokens'].ne(self.padding_idx)
        return copy_loss[non_pad_mask].sum()  # 要取sum，因为label_smoothed_nll_loss中也用的sum

    def compute_loss(self, model, net_output, sample, reduce=True):
        """
        不能在对象上覆盖，要在类上覆盖。
        """
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        # pdb.set_trace()
        # sample['target'].ne(self.padding_idx)  # TODO: mask in copy_dist
        if model.args.sag_type:
            loss += self.get_guidance_loss(net_output, sag_type=model.args.sag_type,
                                           dist_type=model.args.copy_dist_type,
                                           dumping_factor=model.args.dumping_factor)

        if model.args.coverage:
            pass
            # loss += self.coverage_loss()

        return loss, nll_loss


def coverage_loss(epsilon=0.5):
    """ Get to The Point: Summarization with Pointer-Generator Networks
    penalty for repeated words
    See more:
        https://github.com/tensorflow/tensor2tensor/issues/369
        http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html
    """
    pass
