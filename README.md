# Mask2Former + Intra-Batch Supervision
## [[Project page](https://ddegeus.github.io/intra-batch-supervision/)] [[Paper](https://openaccess.thecvf.com/content/WACV2023/html/de_Geus_Intra-Batch_Supervision_for_Panoptic_Segmentation_on_High-Resolution_Images_WACV_2023_paper.html)]

Code for 'Intra-Batch Supervision for Panoptic Segmentation on High-Resolution Images', Daan de Geus and Gijs Dubbelman, WACV 2023.

This code applies Intra-Batch Supervision to [Mask2Former](https://arxiv.org/abs/2112.01527), and is built upon the [official Mask2Former code](https://github.com/facebookresearch/Mask2Former/).

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

* See [Preparing Datasets for Mask2Former](datasets/README.md).
* See [Getting Started with Mask2Former](GETTING_STARTED.md).
* To prepare the datasets for our crop sampling, run these two commands:
  * `python mask2former/data/datasets/prepare_cityscapes_sampling.py`
  * `python mask2former/data/datasets/prepare_mapillary_sampling.py`

  
## Results
Results and models on Cityscapes. 

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Crop sampling</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Iters</th>
<th valign="bottom">PQ</th>
<th valign="bottom">PQ_th</th>
<th valign="bottom">PQ_st</th>
<th valign="bottom">Acc_th</th>
<th valign="bottom">Prec_th</th>
<th valign="bottom">config</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left">Mask2Former</td>
<td align="center">no</td>
<td align="center">R50</td>
<td align="center">90k</td>
<td align="center">62.1</td>
<td align="center">55.2</td>
<td align="center">67.2</td>
<td align="center">87.1</td>
<td align="center">93.3</td>
<td align="center"><a href="configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml">config</a>
<td align="center">TBD</td>
</tr>
<tr><td align="left">Mask2Former + IBS </td>
<td align="center">yes</td>
<td align="center">R50</td>
<td align="center">90k</td>
<td align="center">62.4</td>
<td align="center">55.7</td>
<td align="center">67.3</td>
<td align="center">87.6</td>
<td align="center">94.1</td>
<td align="center"><a href="configs/cityscapes/panoptic-segmentation/maskformer2_R50_bs16_90k_ibs_cropsampling.yaml">config</a></td>
<td align="center">TBD</td>
</tr>
</tbody></table>

Results and models on Mapillary Vistas.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Crop sampling</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Iters</th>
<th valign="bottom">PQ</th>
<th valign="bottom">PQ_th</th>
<th valign="bottom">PQ_st</th>
<th valign="bottom">Acc_th</th>
<th valign="bottom">Prec_th</th>
<th valign="bottom">config</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left">Mask2Former</td>
<td align="center">no</td>
<td align="center">R50</td>
<td align="center">300k</td>
<td align="center">41.5</td>
<td align="center">33.3</td>
<td align="center">52.4</td>
<td align="center">71.7</td>
<td align="center">78.8</td>
<td align="center"><a href="configs/mapillary-vistas/panoptic-segmentation/maskformer_R50_bs16_300k.yaml">config</a>
<td align="center">TBD</td>
</tr>
<tr><td align="left">Mask2Former + IBS </td>
<td align="center">yes</td>
<td align="center">R50</td>
<td align="center">300k</td>
<td align="center">42.2</td>
<td align="center">34.9</td>
<td align="center">52.0</td>
<td align="center">75.7</td>
<td align="center">84.1</td>
<td align="center"><a href="configs/mapillary-vistas/panoptic-segmentation/maskformer_R50_bs16_300k_ibs_cropsampling.yaml">config</a></td>
<td align="center">TBD</td>
</tr>
</tbody></table>



## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This code builds upon the [official Mask2Former code](https://github.com/facebookresearch/Mask2Former/). The majority of Mask2Former is licensed under a [MIT License](LICENSE).

However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## <a name="Citing"></a>Citing us

Please consider citing our work if it is useful for your research.

```
@inproceedings{degeus2023ibs,
  title={Intra-Batch Supervision for Panoptic Segmentation on High-Resolution Images},
  author={{de Geus}, Daan and Dubbelman, Gijs},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2023}
}
```

If you use Mask2Former in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please also refer to the original Mask2Former paper.

```BibTeX
@inproceedings{cheng2022mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```


## Acknowledgement

Code is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former/), which is largely based on MaskFormer (https://github.com/facebookresearch/MaskFormer).
