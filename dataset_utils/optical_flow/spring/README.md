# 1. Download the dataset

Download the dataset files from [here](https://spring-benchmark.org/) and put them into the `DATASETS/Optical_Flow_DATASETS/Spring` folder.

# 2. Unzip the dataset

```
cd DATASETS/Optical_Flow_DATASETS/Spring
unzip train_frame_left.zip
unzip train_flow_FW_left.zip
unzip train_flow_BW_left.zip
unzip test_frame_left.zip
```
The unziped files should be put into the `DATASETS/Optical_Flow_DATASETS/Spring/spring` folder.

# 3. Prepare the dataset

## (1) resize the optical flow images to h x w = 1080 x 1920

The original size of the optical flow is 2160 x 3840, but the size of the image is 1080 x 1920, so we need to resize the flow image to 1080 x 1920. (Note: the values ​​in the original flow match the values ​​of the original image, so after resizing, we do not need to scale the flow values ​​to "1/2" of the original), but we need to perform proportional scaling in subsequent preprocessing.

```
cd DATASETS/Optical_Flow_DATASETS/Spring
python resize_flows.py
```

The resized process will be saved in the original folder, but the original file will be saved in a folder with the suffix "_ori". The prefix is ​​the same as the original folder.

## (2) Compose the forward and backward optical flow (and, warp the start and end images)

The original motions in Spring Dataset are too small, so we need to compose the forward and backward optical flow to increase the motion magtiude.

And then, we need to warp the start and end images to check the correctness of the composed optical flow.

Also, we saved the valid mask in composition and warping, and the content mask and the adjusted weight sum map in forward warping.

```
cd DATASETS/Optical_Flow_DATASETS/Spring
python compose_and_warp.py
```

The composed flows and warped images will be saved in the `DATASETS/Optical_Flow_DATASETS/Spring/spring/preprocessed_h(xxx)w(xxx)` folder.

## (3) Bidirectional consistency check

To get the occlusion mask, we operate the forward check.
To get the disocclusion mask, we operate the backward check.
```
cd DATASETS/Optical_Flow_DATASETS/Spring
python check_consistency.py
```

The results will be saved in the `DATASETS/Optical_Flow_DATASETS/Spring/spring/preprocessed_h(xxx)w(xxx)/consistency_check` folder.

