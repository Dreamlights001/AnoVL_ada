25-05-18 13:03:51.378 - INFO: data_path: D:\AnoCLIP\datasets\MVTec
25-05-18 13:03:51.378 - INFO: save_path: ./results/mvtec/zero_shot_vis_promt
25-05-18 13:03:51.379 - INFO: config_path: ./open_clip/model_configs/ViT-B-16-plus-240.json
25-05-18 13:03:51.379 - INFO: dataset: mvtec
25-05-18 13:03:51.379 - INFO: model: ViT-B-16-plus-240
25-05-18 13:03:51.379 - INFO: pretrained: laion400m_e32
25-05-18 13:03:51.379 - INFO: features_list: [3, 6, 9, 12]
25-05-18 13:03:51.379 - INFO: image_size: 240
25-05-18 13:03:51.379 - INFO: mode: zero_shot
25-05-18 13:03:51.379 - INFO: adapter: True
25-05-18 13:03:51.379 - INFO: epoch: 5
25-05-18 13:03:51.379 - INFO: seed: 111
25-05-18 13:15:53.030 - INFO: 
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   pr_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| bottle     |       92.3 |    53.3 |    55.9 |    80   |       94.8 |    96   |    98.6 |
| cable      |       78.2 |    30   |    23.7 |    71.8 |       83   |    82.4 |    88.6 |
| capsule    |       88   |    12.9 |     7.5 |    66.3 |       79.5 |    91.1 |    95.1 |
| carpet     |       98.9 |    58.8 |    59   |    93.9 |       99.6 |    99.4 |    99.9 |
| grid       |       94.4 |    22.8 |    14.4 |    85.6 |       97.5 |    96.4 |    99.2 |
| hazelnut   |       94.9 |    35.1 |    28.6 |    83.8 |       92.7 |    89.4 |    96.3 |
| leather    |       99   |    39.6 |    35.6 |    95.6 |      100   |   100   |   100   |
| metal_nut  |       76.6 |    34   |    26.5 |    61.4 |       95.1 |    95.7 |    98.9 |
| pill       |       82.7 |    21.6 |    16.5 |    76.6 |       84.6 |    91.6 |    96.8 |
| screw      |       90.7 |    13.2 |     4.3 |    64.5 |       74.3 |    86.8 |    90.4 |
| tile       |       95.8 |    63.6 |    61   |    80.6 |       99.9 |    99.4 |   100   |
| toothbrush |       93.5 |    27.5 |    17.8 |    79.6 |       86.9 |    89.3 |    93.9 |
| transistor |       74.8 |    34.9 |    25.9 |    50.7 |       86.2 |    76.4 |    85.2 |
| wood       |       95.3 |    56.2 |    60.4 |    75.4 |       97.7 |    96.8 |    99.3 |
| zipper     |       93.2 |    38.4 |    26.3 |    78.4 |       88.3 |    91.7 |    96.6 |
| mean       |       89.9 |    36.1 |    30.9 |    76.3 |       90.7 |    92.2 |    95.9 |
