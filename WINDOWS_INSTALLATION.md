# [FILM](https://github.com/google-research/frame-interpolation): Windows Installation Instructions

## Anaconda Python 3.9 (Optional)

#### Install Anaconda3 Python3.9
* Go to [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual) and click the "Download" button.
* Download the Windows [64-Bit](https://repo.anaconda.com/archive/Anaconda3-2021.11-Windows-x86_64.exe) or [32-bit](https://repo.anaconda.com/archive/Anaconda3-2021.11-Windows-x86.exe) Graphical Installer, depending on your system needs.
* Run the downloaded (`.exe`) file to begin the installation.
* (Optional) Check the "Add Anaconda3 to my PATH environment variable". You may get a 'red text' warning of its implications, you may ignore it for this setup.

#### Create a new Anaconda virtual environment
* Open a new Terminal
* Type the following command:
```
conda create -n frame_interpolation pip python=3.9
```
* The above command will create a new virtual environment with the name `frame_interpolation`

#### Activate the Anaconda virtual environment
* Activate the newly created virtual environment by typing in your terminal (Command Prompt or PowerShell)
```
conda activate frame_interpolation
```
* Once activated, your terminal should look like:
```
(frame_interpolation) <present working directory> >
```

## NVIDIA GPU Support
#### Install CUDA Toolkit
* Go to [https://developer.nvidia.com/cuda-11.2.1-download-archive](https://developer.nvidia.com/cuda-11.2.1-download-archive) and select your `Windows`.
* Download and install `CUDA Tookit 11.2.1`.
* Additional CUDA installation information available [here](https://docs.nvidia.com/cuda/archive/11.2.2/cuda-installation-guide-microsoft-windows/index.html).

#### Install cuDNN
* Go to [https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download).
* Create a user profile (if needed) and login.
* Select `cuDNN v8.1.0 (January 26th, 2021), for CUDA 11.0,11.1 and 11.2`.
* Download [cuDNN Library for Widnows (x86)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.0.77/11.2_20210127/cudnn-11.2-windows-x64-v8.1.0.77.zip). 
* Extract the contents of the zipped folder (it contains a folder named `cuda`) into `<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\`. `<INSTALL_PATH>` points to the installation directory specified during CUDA Toolkit installation. By default, `<INSTAL_PATH> = C:\Program Files`.

#### Environment Setup
* Add the following paths to your 'Advanced System Settings' > 'Environment Variables ...' > Edit 'Path', and add:
    * <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
    * <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
    * <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include
    * <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64
    * <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\cuda\bin

#### Verify Installation
* Open a **new** terminal and type `conda activate frame_interpolation`.
* Install (temporarily) tensorflow and run a simple operation, by typing:
```
pip install --ignore-installed --upgrade tensorflow==2.6.0
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
* You should see success messages: 'Created device /job:localhost/replica:0/task:0/device:GPU:0'.

## FILM Installation
* Get Frame Interpolation source codes
```
git clone https://github.com/google-research/frame-interpolation
cd frame-interpolation
```
* Install dependencies
```
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```
* Download pre-traned models, detailed [here](https://github.com/google-research/frame-interpolation#pre-trained-models).

## Running the Codes
* One mid-frame interpolation. Note: `python3` may not be recognized in Windows, so simply drop `3` as below.
```
python -m eval.interpolator_test --frame1 photos\one.png --frame2 photos\two.png --model_path <pretrained_models>\film_net\Style\saved_model --output_frame photos\output_middle.png
```

* Large resolution mid-frame interpolation: Set `block_height` and `--block_width` to subdivide along the height and width to create patches, where the interpolator will be run iteratively, and the resulting interpolated mid-patches will be reconstructed into a final mid-frame. In the example below, will create and run on 4 patches (2*2).
```
python -m eval.interpolator_test --frame1 photos\one.png --frame2 photos\two.png --block_height 2 --block_wdith 2 --model_path <pretrained_models>\film_net\Style\saved_model --output_frame photos\output_middle.png
```
* Many in-between frames interpolation
```
python -m eval.interpolator_cli --pattern "photos" --model_path <pretrained_models>\film_net\Style\saved_model --times_to_interpolate 6 --output_video
```

## Acknowledgments

This windows installation guide is heavily based on [tensorflow-object-detection-api-tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) .
