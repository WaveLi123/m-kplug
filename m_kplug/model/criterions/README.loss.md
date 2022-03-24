

## 通常的训练指标

backtranslation: (这里的ppl怎么这么低？？)
2020-02-07 13:21:07 | INFO | train | {"epoch": 1, "train_loss": "11.049", "train_nll_loss": "10.399", "train_ppl": "1350.12", "train_wps": "2.39679e+06", "train_ups": "5.95", "train_wpb": "402992", "train_bsz": "14123.2", "train_num_updates": "362", "train_lr": "9.05909e-05", "train_gnorm": "1.682", "train_clip": "0", "train_oom": "0", "train_loss_scale": "20", "train_train_wall": "69", "train_wall": "119"}
2020-02-07 13:35:34 | INFO | valid | {"epoch": 10, "valid_loss": "4.033", "valid_nll_loss": "2.269", "valid_ppl": "4.82", "valid_wps": "3.14724e+06", "valid_wpb": "373365", "valid_bsz": "13099", "valid_num_updates": "3645", "valid_best_loss": "4.033"}


mass_pretrain （ppl有点高）
2020-07-31 08:40:22 | INFO | valid | epoch 370 | valid on 'valid' subset | loss 3.819 | ppl 14.12 | wps 91452.5 | wpb 8664 | bsz 120.3 | num_updates 232011 | best_loss 3.765



mass_finetune


jd_pretrain


jd_finetune


