# DailyLife

## 5-fold CV results

Binary classification (low vs. high) with 2,181 1-min segments of `['gsr', 'bpm', 'rri', 'temp']`.

### Arousal

```console
$ python baseline.py -r ~/data/dailyLife2/datasets/60s -e ~/data/dailyLife2/metadata/esms_activity.csv -s 1 -t arousal --cv 'kfold' --splits 5 --shuffle --gpu
```

| Metric   |   Random |   Majority |   Class ratio |   Gaussian NB |   XGBoost |
|:---------|---------:|-----------:|--------------:|--------------:|----------:|
| acc.     | 0.501611 |   0.52132  |      0.499783 |      0.553404 |  0.55662  |
| auroc    | 0.5      |   0.5      |      0.499221 |      0.577678 |  0.586021 |
| bacc.    | 0.501703 |   0.5      |      0.499221 |      0.557775 |  0.555669 |
| f1       | 0.510292 |   0.685352 |      0.515729 |      0.513229 |  0.575862 |

### Valence

```console
$ python baseline.py -r ~/data/dailyLife2/datasets/60s -e ~/data/dailyLife2/metadata/esms_activity.csv -s 1 -t valence --cv 'kfold' --splits 5 --shuffle --gpu
```

| Metric   |   Random |   Majority |   Class ratio |   Gaussian NB |   XGBoost |
|:---------|---------:|-----------:|--------------:|--------------:|----------:|
| acc.     | 0.491061 |   0.786795 |      0.661163 |      0.746965 |  0.75929  |
| auroc    | 0.5      |   0.5      |      0.498554 |      0.521622 |  0.568744 |
| bacc.    | 0.48922  |   0.5      |      0.498554 |      0.510779 |  0.521717 |
| f1       | 0.603419 |   0.880677 |      0.783953 |      0.847594 |  0.859448 |