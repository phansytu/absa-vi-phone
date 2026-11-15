# So sánh Baseline V2C vs PhoBERT

*Số seed PhoBERT: 3 (seeds: [42, 123, 456])*

| Model | Joint F1 | Aspect Micro | Aspect Macro | Sentiment (gold) | Empty Acc |
| --- | --- | --- | --- | --- | --- |
| Baseline V2C | 0.7487 | 0.8655 | 0.8387 | 0.7196 | 0.3333 |
| PhoBERT (n=3 seeds) | 0.8079 ± 0.0023 | 0.8986 ± 0.0016 | 0.8020 ± 0.0070 | 0.7661 ± 0.0034 | 0.3333 ± 0.0000 |


## Delta (PhoBERT - Baseline)

- **Joint F1**: +0.0593
- **Aspect Micro**: +0.0331
- **Aspect Macro**: -0.0367
- **Sentiment (gold)**: +0.0466
- **Empty Acc**: 0.0000
