# Beyond Myopia: Learning from Positive and Unlabeled Data through Holistic Predictive Trends [NeurIPS 2023]
## Method
We demonstrate the effectiveness of the proposed resampling strategy. It is also observed that predictive trends for each example can serve as an important metric for discriminating the categories of unlabeled data, providing a novel perspective for PUL. After that, we propose a new measure, trend score, which is proved to be asymptotically unbiased in the change of predicting scores. We then introduce a modified version of Fisher’s Natural Break with lower time complexity to identify statistically significant partitions. This process does not require additional tuning efforts and prior assumptions.
### Negative trend
<p align="center">
  <img src="pics/negative_trend_00.png" width="700">
</p>

### Positive trend
<p align="center">
  <img src="pics/positive_trend_00.png" width="700">
</p>

### Trend Statistics
<p align="center">
  <img src="pics/trend_statistic_00.png" width="1200">
</p>
### Note
The provided code is much implemented on the simplified version of TS score, as it's a bit faster and achieves similar results.
### Running
```
sh ./run.sh
```
All best hyperparameters are reported in run.sh
## Reference
@article{wang2023beyond,\
  title={Beyond Myopia: Learning from Positive and Unlabeled Data through Holistic Predictive Trends},\
  author={Wang, Xinrui and Wan, Wenhai and Geng, Chuanxin and LI, Shaoyuan and Chen, Songcan},\
  journal={arXiv preprint arXiv:2310.04078},\
  year={2023}\
}
