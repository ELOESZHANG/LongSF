# LongSF
This is the official version of LongSF. Ursus-Mamba is mainly used for multimodal 3D object detection with SSMs, and we conduct experiments on nuScenes and KITTI datasets.

## Evaluation results on the KITTI benchmark
<figure>
  <img src="./tracking/EVALUATION RESULTS.png" alt="table">
  <figcaption style="text-align: center;"></figcaption>
</figure>

## Test
1. Please download the LSOTB-TIR evaluation dataset, PTB-TIR dataset, and VOT-TIR2015 dataset.
2. Configure the path in  `lib/test/evaluation/local.py` and `lib/test/parameter/nlmtrack.py`.
3. Run `tracking/test.py` for testing
4. Evaluation on the [LSOTB-TIR benchmark](https://github.com/QiaoLiuHit/LSOTB-TIR), [PTB-TIR benchmark](https://github.com/QiaoLiuHit/PTB-TIR_Evaluation_toolkit), and [VOT benchmark](https://github.com/votchallenge/toolkit-legacy)

## Train
1. Preparing your training data(GOT-10K dataset, LSOTB-TIR training dataset)
2. Configure the path in `lib/train/admin/local.py`
3. Run `lib/train/run_training.py`

