
入口：@register_criterion('masked_lm')


1. 调用前向网络
2. 计算loss


```


def forward(self, model, sample

    logits = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
    targets = model.get_targets(sample, [logits])

    loss = f(logits, targets)

```


--criterion masked_lm_loss 
--criterion legacy_masked_lm_loss 

## 入口


```
task = tasks.setup_task(args)
task.load_dataset(valid_sub_split, combine=False, epoch=1)
model = task.build_model(args)
criterion = task.build_criterion(args)  # 这里会在args中查找匹配Criterion构造函数的参数。
```

