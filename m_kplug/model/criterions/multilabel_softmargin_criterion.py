"""

https://github.com/Megvii-Nanjing/ML-GCN/blob/master/demo_voc2007_gcn.py#L53


跟BCELoss的区别？ https://discuss.pytorch.org/t/what-is-the-difference-between-bcewithlogitsloss-and-multilabelsoftmarginloss/14944
"""

from torch import nn

criterion = nn.MultiLabelSoftMarginLoss()