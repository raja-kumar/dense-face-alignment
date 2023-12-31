# dense-face-alignment

Most of the existing face-alignments detects the standard 68 face landmarks. However, there is a need of dense facial landmarks for tasks such 3d face reconstruction, face recognition etc. [DAD-3DHeads](https://github.com/PinataFarms/DAD-3DHeads) proposed a dense 2d landmark detector based on 3D face reconstruction. However, their performance deteriorates when face is only a part of the image. see figure below. To improve on this, I used mediapipe facial detector to detect the face, use the detected face to DADNet and then project back the detected landmarks in the original image coordinate.


<img src="./images/with_without_detector.png" width="512" height="320">      <img src="./images/sample.jpeg" alt="dense landmarks sample" width="256" height="320">

## Dependencies

`
pip install -r requirements.txt
`

Note: if you are using mac silicon replace mediapipe with mediapipe-silicon in requirements.txt

## How to replicate the results

`
python run.py -h
`

sample command

`
python run.py --data_path sample_input --num_points 445 
`

sample command for custom points

`
python run.py --data_path sample_input --custom_index model_training/model/static/face_keypoints/ids1.npy
`

## Citation

If you found this work useful for your research, please consider citing our paper

```
@inproceedings{kumar2023disjoint,
  title={Disjoint Pose and Shape for 3D Face Reconstruction},
  author={Kumar, Raja and Luo, Jiahao and Pang, Alex and Davis, James},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages={3115--3125},
  year={2023}
}
```
## Attributions
this repo is built on top of base code from [DAD-3DHeads](https://github.com/PinataFarms/DAD-3DHeads)
