# FILM: Frame Interpolation for Large Motion

### [Website](https://film-net.github.io/) | [Paper](https://arxiv.org/pdf/2202.04901.pdf) | [Google AI Blog](https://ai.googleblog.com/2022/10/large-motion-frame-interpolation.html) | [Tensorflow Hub Colab](https://www.tensorflow.org/hub/tutorials/tf_hub_film_example) | [YouTube](https://www.youtube.com/watch?v=OAD-BieIjH4) <br>

The official Tensorflow 2 implementation of our high quality frame interpolation neural network. We present a unified single-network approach that doesn't use additional pre-trained networks, like optical flow or depth, and yet achieve state-of-the-art results. We use a multi-scale feature extractor that shares the same convolution weights across the scales. Our model is trainable from frame triplets alone. <br>

[FILM: Frame Interpolation for Large Motion](https://arxiv.org/abs/2202.04901) <br />
[Fitsum Reda](https://fitsumreda.github.io/)<sup>1</sup>, [Janne Kontkanen](https://scholar.google.com/citations?user=MnXc4JQAAAAJ&hl=en)<sup>1</sup>, [Eric Tabellion](http://www.tabellion.org/et/)<sup>1</sup>, [Deqing Sun](https://deqings.github.io/)<sup>1</sup>, [Caroline Pantofaru](https://scholar.google.com/citations?user=vKAKE1gAAAAJ&hl=en)<sup>1</sup>, [Brian Curless](https://homes.cs.washington.edu/~curless/)<sup>1,2</sup><br />
<sup>1</sup>Google Research, <sup>2</sup>University of Washington<br />
In ECCV 2022.

![A sample 2 seconds moment.](https://github.com/googlestaging/frame-interpolation/blob/main/moment.gif)
FILM transforms near-duplicate photos into a slow motion footage that look like it is shot with a video camera.

## Web Demo

Integrated into [Hugging Face Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/johngoad/frame-interpolation)

Try the interpolation model with the replicate web demo at 
[![Replicate](https://replicate.com/google-research/frame-interpolation/badge)](https://replicate.com/google-research/frame-interpolation)

Try FILM to interpolate between two or more images with the PyTTI-Tools at [![PyTTI-Tools:FILM](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/pytti-tools/frame-interpolation/blob/main/PyTTI_Tools_FiLM-colab.ipynb#scrollTo=-7TD7YZJbsy_)

An alternative Colab for running FILM on arbitrarily more input images, not just on two images, [![FILM-Gdrive](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NuaPPSvUhYafymUf2mEkvhnEtpD5oihs)

## Change Log
* **Nov 28, 2022**: Upgrade `eval.interpolator_cli` for **high resolution frame interpolation**. `--block_height` and `--block_width` determine the total number of patches (`block_height*block_width`) to subdivide the input images. By default, both arguments are set to 1, and so no subdivision will be done.
* **Mar 12, 2022**: Support for Windows, see [WINDOWS_INSTALLATION.md](https://github.com/google-research/frame-interpolation/blob/main/WINDOWS_INSTALLATION.md).
* **Mar 09, 2022**: Support for **high resolution frame interpolation**. Set `--block_height` and `--block_width` in `eval.interpolator_test` to extract patches from the inputs, and reconstruct the interpolated frame from the iteratively interpolated patches.

## Installation

*   Get Frame Interpolation source codes

```
git clone https://github.com/google-research/frame-interpolation
cd frame-interpolation
```

*   Optionally, pull the recommended Docker base image

```
docker pull gcr.io/deeplearning-platform-release/tf2-gpu.2-6:latest
```

* If you do not use Docker, set up your NVIDIA GPU environment with:
    * [Anaconda Python 3.9](https://www.anaconda.com/products/individual)
    * [CUDA Toolkit 11.2.1](https://developer.nvidia.com/cuda-11.2.1-download-archive)
    * [cuDNN 8.1.0](https://developer.nvidia.com/rdp/cudnn-download)

*   Install frame interpolation dependencies

```
pip3 install -r requirements.txt
sudo apt-get install -y ffmpeg
```

### See [WINDOWS_INSTALLATION](https://github.com/google-research/frame-interpolation/blob/main/WINDOWS_INSTALLATION.md) for Windows Support

## Pre-trained Models

*   Create a directory where you can keep large files. Ideally, not in this
    directory.

```
mkdir -p <pretrained_models>
```

*   Download pre-trained TF2 Saved Models from
    [google drive](https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy?usp=sharing)
    and put into `<pretrained_models>`.

The downloaded folder should have the following structure:

```
<pretrained_models>/
├── film_net/
│   ├── L1/
│   ├── Style/
│   ├── VGG/
├── vgg/
│   ├── imagenet-vgg-verydeep-19.mat
```

## Running the Codes

The following instructions run the interpolator on the photos provided in
'frame-interpolation/photos'.

### One mid-frame interpolation

To generate an intermediate photo from the input near-duplicate photos, simply run:

```
python3 -m eval.interpolator_test \
   --frame1 photos/one.png \
   --frame2 photos/two.png \
   --model_path <pretrained_models>/film_net/Style/saved_model \
   --output_frame photos/output_middle.png
```

This will produce the sub-frame at `t=0.5` and save as 'photos/output_middle.png'.

### Many in-between frames interpolation

It takes in a set of directories identified by a glob (--pattern). Each directory
is expected to contain at least two input frames, with each contiguous frame
pair treated as an input to generate in-between frames. Frames should be named such that when sorted (naturally) with `natsort`, their desired order is unchanged.

```
python3 -m eval.interpolator_cli \
   --pattern "photos" \
   --model_path <pretrained_models>/film_net/Style/saved_model \
   --times_to_interpolate 6 \
   --output_video
```

You will find the interpolated frames (including the input frames) in
'photos/interpolated_frames/', and the interpolated video at
'photos/interpolated.mp4'.

The number of frames is determined by `--times_to_interpolate`, which controls
the number of times the frame interpolator is invoked. When the number of frames
in a directory is `num_frames`, the number of output frames will be
`(2^times_to_interpolate+1)*(num_frames-1)`.

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
datasets.create_<dataset_name>_tfrecord --help` for more information.

For example, run the command below to create a TFRecord for the Middlebury-other
dataset. Download the [images](https://vision.middlebury.edu/flow/data) and point `--input_dir` to the unzipped folder path.

```
python3 -m datasets.create_middlebury_tfrecord \
  --input_dir=<root folder of middlebury-other> \
  --output_tfrecord_filepath=<output tfrecord filepath> \
  --num_shards=3
```

The above command will output a TFRecord file with 3 shards as `<output tfrecord filepath>@3`.

## Training

Below are our training gin configuration files for the different loss function:

```
training/
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
python3 -m training.train \
   --gin_config training/config/<config filename>.gin \
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
python3 -m training.build_saved_model_cli \
   --base_folder <base folder of training sessions> \
   --label <the name of the run>
```

*   By default, a SavedModel is created when the training loop ends, and it will be saved at
    `<base_folder>/<label>/saved_model`.

## Evaluation on Benchmarks

Below, we provided the evaluation gin configuration files for the benchmarks we
have considered:

```
eval/
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
python3 -m eval.eval_cli \
   --gin_config eval/config/<eval_dataset>.gin \
   --model_path <pretrained_models>/film_net/L1/saved_model
```

The above command will produce the PSNR and SSIM scores presented in the paper.

## Citation

If you find this implementation useful in your works, please acknowledge it
appropriately by citing:

```
@inproceedings{reda2022film,
 title = {FILM: Frame Interpolation for Large Motion},
 author = {Fitsum Reda and Janne Kontkanen and Eric Tabellion and Deqing Sun and Caroline Pantofaru and Brian Curless},
 booktitle = {European Conference on Computer Vision (ECCV)},
 year = {2022}
}
```

```
@misc{film-tf,
  title = {Tensorflow 2 Implementation of "FILM: Frame Interpolation for Large Motion"},
  author = {Fitsum Reda and Janne Kontkanen and Eric Tabellion and Deqing Sun and Caroline Pantofaru and Brian Curless},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/google-research/frame-interpolation}}
}
```

## Acknowledgments

We would like to thank Richard Tucker, Jason Lai and David Minnen. We would also
like to thank Jamie Aspinall for the imagery included in this repository.

## Coding style

*   2 spaces for indentation
*   80 character line length
*   PEP8 formatting

## Disclaimer

This is not an officially supported Google product.
