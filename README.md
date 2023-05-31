# 2D Skeleton-based Action Recognition

For this project, we started from the code of the paper Revisiting Skeleton-based Action Recognition (Duan et al., 2022). For each frame in a video, they first use a two-stage pose estimator (detection + pose estimation) for 2D human pose extraction. Then they stack heatmaps of joints or limbs along the temporal dimension and apply pre-processing to the generated 3D heatmap volumes. Finally, they use a 3D-CNN to classify the 3D heatmap volumes. The new architecture they proposed is called PoseConv3d. For further information, here is their Github repository [PYSKL repository](https://github.com/kennymckormick/pyskl.git).

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/142995620-21b5536c-8cda-48cd-9cb9-50b70cab7a89.png" width=80%/>
</div>

## Contribution
The method developed in the paper of Duan et al. achieves state-of-the-art results. However, it is computationally heavy as we experienced when training their model in scitas. Indeed, since they use heatmaps, instead of skeleton keypoints, they have an input shape of (batch_size, #joints, #frames, height, width).

Our idea is to get rid of the heatmap to optimize the computation costs. Instead we use the 2D skeleton data as they are without any modifications to do the classification. This gives an input shape of (batch_size, #persons, #joints, #frames, #dimensions), #dimensions is 2 for 2D keypoints and 3 for 3D keypoints. We use nearly the same architecture and adapt the stride, kernel size, and padding of 3D-conv and avg pool. The network was not learning at all, so we try different input shape, we added normalization of the keypoints coordinate.

Talk bout grayscale images with (batch_size, 1, 32, height, width)

## Experimental Setup
Here we will present the experiments we conducted and the evaluation metrics we use to quatify the results.
### Experiments



### Evaluation metrics
Accuracy, precision and recall, F1 score and confusion matrix.


## Dataset
The dataset used for this project is called NTURGB+D and was developed by Shahroudy et al. (2016). It contains more than 56,000 video samples collected from 40 distinct subjects. This dataset contains 60 different action classes including daily, mutual, and health-related actions.

Since our project is about 2D Skeleton-based Action Recognition, we use a pre-processed 2D skeletons dataset provided by PYSKL. It is an annotated dataset
- Original Dataset (Shahroudy et al., 2016): [NTU-RGB+D](https://arxiv.org/abs/1604.02808)
- Pre-processed dataset (PYSKL): [NTU-RGB+D 2D SkeletonS download (pickle file)](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl)

<div id="wrapper" align="center">
<figure>
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="520px">&emsp;
  <p style="font-size:1.2vw;">Skeleton-base Action Recognition Results on NTU-RGB+D (PYSKL)</p>
</figure>
</div>

The content of a pickle file is a dictionary with two fields: `split` and `annotations`

1. Split: The value of the `split` field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
2. Annotations: The value of the `annotations` field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
   1. `frame_dir` (str): The identifier of the corresponding video.
   2. `total_frames` (int): The number of frames in this video.
   3. `img_shape` (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
   4. `original_shape` (tuple[int]): Same as `img_shape`.
   5. `label` (int): The action label.
   6. `keypoint` (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as `total_frames`); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
   7. `keypoint_score` (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.



## Results



## Conclusion

## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Demo



## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash 
# Testing
bash 
```








## Citation
