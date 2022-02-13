# FILM: Frame Interpolation for Large Scene Motion

### [Project](https://film-net.github.io/) | [Paper](https://arxiv.org/pdf/2202.04901.pdf) | [YouTube](https://www.youtube.com/watch?v=OAD-BieIjH4) | [Benchmark Scores](https://github.com/google-research/frame-interpolation) <br>

Tensorflow 2 implementation of our high quality frame interpolation neural network. We present a unified single-network approach that doesn't use additional pre-trained networks, like optical flow or depth, and yet achieve state-of-the-art results. We use a multi-scale feature extractor that shares the same convolution weights across the scales. Our model is trainable from frame triplets alone. <br>

[FILM: Frame Interpolation for Large Motion](https://arxiv.org/abs/2202.04901) <br />
[Fitsum Reda](https://scholar.google.com/citations?user=quZ_qLYAAAAJ&hl=en), [Janne Kontkanen](https://scholar.google.com/citations?user=MnXc4JQAAAAJ&hl=en), [Eric Tabellion](http://www.tabellion.org/et/), [Deqing Sun](https://deqings.github.io/), [Caroline Pantofaru](https://scholar.google.com/citations?user=vKAKE1gAAAAJ&hl=en), [Brian Curless](https://homes.cs.washington.edu/~curless/)<br />
Google Research <br />
Technical Report 2022.

![A sample 2 seconds moment.](https://github.com/googlestaging/frame-interpolation/blob/main/moment.gif)
FILM transforms near-duplicate photos into a slow motion footage that look like it is shot with a video camera.

## Web Demo

Try the interpolation model with the replicate web demo at 
[![Replicate](https://replicate.com/google-research/frame-interpolation/badge)](https://replicate.com/google-research/frame-interpolation)

## Installation

*   Get Frame Interpolation source codes

```
> git clone https://github.com/google-research/frame-interpolation frame_interpolation
```

*   Optionally, pull the recommended Docker base image

```
> docker pull gcr.io/deeplearning-platform-release/tf2-gpu.2-6:latest
```

*   Install dependencies

```
> pip install -r frame_interpolation/requirements.txt
> apt-get install ffmpeg
```

## Pre-trained Models

*   Create a directory where you can keep large files. Ideally, not in this
    directory.

```
> mkdir <pretrained_models>
```

*   Download pre-trained TF2 Saved Models from
    [google drive](https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy?usp=sharing)
    and put into `<pretrained_models>`.

The downloaded folder should have the following structure:

```
pretrained_models/
├── film_net/
│   ├── L1/
│   ├── VGG/
│   ├── Style/
├── vgg/
│   ├── imagenet-vgg-verydeep-19.mat
```

## Running the Codes

The following instructions run the interpolator on the photos provided in
frame_interpolation/photos.

### One mid-frame interpolation

To generate an intermediate photo from the input near-duplicate photos, simply run:

```
> python3 -m frame_interpolation.eval.interpolator_test \
     --frame1 frame_interpolation/photos/one.png \
     --frame2 frame_interpolation/photos/two.png \
     --model_path <pretrained_models>/film_net/Style/saved_model \
     --output_frame frame_interpolation/photos/middle.png \
```

This will produce the sub-frame at `t=0.5` and save as
'frame_interpolation/photos/middle.png'.

### Many in-between frames interpolation

Takes in a set of directories identified by a glob (--pattern). Each directory
is expected to contain at least two input frames, with each contiguous frame
pair treated as an input to generate in-between frames.

```
> python3 -m frame_interpolation.eval.interpolator_cli \
     --pattern "frame_interpolation/photos" \
     --model_path <pretrained_models>/film_net/Style/saved_model \
     --times_to_interpolate 6 \
     --output_video
```

You will find the interpolated frames (including the input frames) in
'frame_interpolation/photos/interpolated_frames/', and the interpolated video at
'frame_interpolation/photos/interpolated.mp4'.

The number of frames is determined by `--times_to_interpolate`, which controls
the number of times the frame interpolator is invoked. When the number of frames
in a directory is 2, the number of output frames will be
`2^times_to_interpolate+1`.

## Datasets

We use [Vimeo-90K](http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip) as
our main training dataset. For quantitative evaluations, we rely on commonly
used benchmark datasets, specifically:

*   [Vimeo-90K](http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip)
*   [Middlebury-Other](https://vision.middlebury.edu/flow/data)
*   [UCF101](https://people.cs.umass.edu/~hzjiang/projects/superslomo/UCF101_results.zip)
*   [Xiph](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark.py)

### Creating a TFRecord

The training and benchmark evaluation scripts expect the frame triplets in the
[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) storage format. <br />

We have included scripts that encode the relevant frame triplets into a
[tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example)
data format, and export to a TFRecord file. <br />

You can use the commands `python3 -m
frame_interpolation.datasets.create_<dataset_name>_tfrecord --help` for more information.

For example, run the command below to create a TFRecord for the Middlebury-other
dataset. Download the [images](https://vision.middlebury.edu/flow/data) and point `--input_dir` to the unzipped folder path.

```
> python3 -m frame_interpolation.datasets.create_middlebury_tfrecord \
    --input_dir=<root folder of middlebury-other> \
    --output_tfrecord_filepath=<output tfrecord filepath>
```

## Training

Below are our training gin configuration files for the different loss function:

```
frame_interpolation/training/
├── config/
│   ├── film_net-L1.gin
│   ├── film_net-VGG.gin
│   ├── film_net-Style.gin
```

To launch a training, simply pass the configuration filepath to the desired
experiment. <br />
By default, it uses all visible GPUs for training. To debug or train
on a CPU, append `--mode cpu`.

```
> python3 -m frame_interpolation.training.train \
     --gin_config frame_interpolation/training/config/<config filename>.gin \
     --base_folder <base folder for all training runs> \
     --label <descriptive label for the run>
```

*   When training finishes, the folder structure will look like this:

```
<base_folder>/
├── <label>/
│   ├── config.gin
│   ├── eval/
│   ├── train/
│   ├── saved_model/
```

### Build a SavedModel

Optionally, to build a
[SavedModel](https://www.tensorflow.org/guide/saved_model) format from a trained
checkpoints folder, you can use this command:

```
> python3 -m frame_interpolation.training.build_saved_model_cli \
     --base_folder <base folder of training sessions> \
     --label <the name of the run>
```

*   By default, a SavedModel is created when the training loop ends, and it will be saved at
    `<base_folder>/<label>/<saved_model>`.

## Evaluation on Benchmarks

Below, we provided the evaluation gin configuration files for the benchmarks we
have considered:

```
frame_interpolation/eval/
├── config/
│   ├── middlebury.gin
│   ├── ucf101.gin
│   ├── vimeo_90K.gin
│   ├── xiph_2K.gin
│   ├── xiph_4K.gin
```

To run an evaluation, simply pass the configuration file of the desired evaluation dataset. <br />
If a GPU is visible, it runs on it.

```
> python3 -m frame_interpolation.eval.eval_cli -- \
     --gin_config frame_interpolation/eval/config/<eval_dataset>.gin \
     --model_path <pretrained_models>/film_net/L1/saved_model
```

The above command will produce the PSNR and SSIM scores presented in the paper.

## Citation

If you find this implementation useful in your works, please acknowledge it
appropriately by citing:

```
@inproceedings{reda2022film,
 title = {Frame Interpolation for Large Motion},
 author = {Fitsum Reda and Janne Kontkanen and Eric Tabellion and Deqing Sun and Caroline Pantofaru and Brian Curless},
 booktitle = {arXiv},
 year = {2022}
}
```

```
@misc{film-tf,
  title = {Tensorflow 2 Implementation of "FILM: Frame Interpolation for Large Scene Motion"},
  author = {Fitsum Reda and Janne Kontkanen and Eric Tabellion and Deqing Sun and Caroline Pantofaru and Brian Curless},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/google-research/frame-interpolation}}
}
```
Contact: Fitsum Reda (fitsum@google.com)

## Acknowledgments

We would like to thank Richard Tucker, Jason Lai and David Minnen. We would also
like to thank Jamie Aspinall for the imagery included in this repository.

## Coding style

*   2 spaces for indentation
*   80 character line length
*   PEP8 formatting

## Disclaimer

This is not an officially supported Google product.
