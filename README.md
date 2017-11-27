# Deep Learning for Computer Vision 
# Technical University Munich - WS 2017

1. Python Setup
2. PyTorch Installation
3. Exercise Download
4. Dataset Download
5. Exercise Submission
6. Remote Access for Jupyter Notebooks
7. Acknowledgments


## 1. Python Setup

Prerequisites:
- Unix system (Linux or MacOS)
- Python version 3.6
- Terminal (e.g. iTerm2 for MacOS)
- Integrated development environment (IDE) (e.g. PyCharm or Sublime Text)

For the following description, we assume that you are using Linux or MacOS and that you are familiar with working from a terminal. The exercises are implemented in Python 3.6.

If you are using Windows, the procedure might slightly vary and you will have to Google for the details. A fellow student of yours compiled this (https://gitlab.lrz.de/yuesong.shen/DL4CV-win) very detailed Windows tutorial for our course. Please keep in mind, that we will not offer any kind of support for its content.

To avoid issues with different versions of Python and Python packages we recommend to always set up a project specific virtual environment. The most common tools for a clean management of Python environments are *pyenv*, *virtualenv* and *Anaconda*.

In this README we provide you with a short tutorial on how to use and setup a *virtuelenv* environment. To this end, install or upgrade *virtualenv*. There are several ways depending on your OS. At the end of the day, we want 

`which virtualenv`

to point to the installed location.

On Ubuntu, you can use: 

`apt-get install python-virtualenv`

Also, installing with pip should work (the *virtualenv* executable should be added to your search path automatically):

`pip3 install virtualenv`

Once *virtualenv* is successfully installed, go to the root directory of the dl4cv repository (where this README.md is located) and execute:

`virtualenv -p python3 --no-site-packages .venv`

Basically, this installs a sandboxed Python in the directory `.venv`. The
additional argument ensures that sandboxed packages are used even if they had
already been installed globally before.

Whenever you want to use this *virtualenv* in a shell you have to first
activate it by calling:

`source .venv/bin/activate`

To test whether your *virtualenv* activation has worked, call:

`which python`

This should now point to `.venv/bin/python3`.

From now on we assume that that you have activated your virtual environment.

Installing required packages:
We have made it easy for you to get started, just call from the dl4cv root directory:

`pip3 install -r requirements.txt`

The exercises are guided via Jupyter Notebooks (files ending with `*.ipynb`). In order to open a notebook dedicate a separate shell to run a Jupyter Notebook server in the dl4cv root directory by executing:

`jupyter notebook`

A browser window which depicts the file structure of directory should open (we tested this with Chrome). From here you can select an exercise directory and one of its exercise notebooks!


## 2. PyTorch installation

In exercise 3 we will introduce the *PyTorch* deep learning framework which provides a research oriented interface with a dynamic computation graph and many predefined, learning-specific helper functions.

Unfortunately, the installation depends on the individual system configuration (OS, Python version and CUDA version) and therefore is not possible with the usual `requirements.txt` file.

Follow the *Get Started* section on the official PyTorch [website](http://pytorch.org/) to choose and install your version.


## 3. Exercise Download

Our exercise is structured with git submodules. At each time we start with a new exercise you have to populate the respective exercise directory. Access to the corresponding repositories will be granted once the new exercise starts. 
You obtain the exercises by first updating the dl4cv root repository:

`git pull origin master`

and then pulling the respective exercise submodule:

`git submodule update --init -- exercise_{1, 2, 3}`


## 4. Dataset Download

To download the datasets required for an exercise, execute the respective download script located in the exercise directory:

`./get_datasets.sh`

You will need ~400MB of disk space.


## 5. Exercise Submission

After completing an exercise you will be submitting trained models to be
automatically evaluated on a test set on our server. To this end, login or register for an account at:

https://vision.in.tum.de/teaching/ws2017/dl4cv/submit

Note that only students, who have registered for this class in TUM Online can
register for an account. This account provides you with temporary credentials to login onto the machines at our chair. In addition to your own computers, you may also use the computers in room 02.05.14 which allow you to train your models on a GPU. Keep in mind, that training on a GPU is considerably faster but not necessarily required for the exercises.

After you have worked through an exercise, your saved models will be in the corresponding `models` subfolder of this exercise. In order to submit the models you execute our submit script:

`./submit_exercise.sh X s999`

where `X={1,2,3}` for the respective exercise and `s999` has to be substituted by your username in our system.

This script uses *rsync* to transfer your code and the models onto our test server and into your user's home directory `~/submit/EX{1, 2, 3}`. Make sure *rsync* is installed on your local machine and don't change the filenames of your models!

Once the models are uploaded to `~/submit/EX{1, 2, 3}`, you can login to the above website, where they can be selected for submission. Note that you have to explicitly submit the files through our web interface, just uploading them to the respective directory is not enough.

You will receive an email notification with the results upon completion of the
evaluation. To make it more fun, you will be able to see a leader board of everyone's (anonymous) scores on the login part of the submission website.

Note that you can re-evaluate your models until the deadline of the current exercise. Whereas the email contains the result of the current evaluation, the entry in the leader board always represents the best score for the respective exercise.


## 6. Remote Access for Jupyter Notebooks

In order to use Jupyter notebooks remotely, i.e. running the notebook in your browser while the actual code is evaluated on a host machine, you first need to start a headless Jupyter notebook server on a specific port (e.g. 7777) on the host machine:

`jupyter notebook --no-browser --port=7000`

You then establish a ssh tunnel from the host machine (e.g. atbeetz21) to a port on your machine (e.g 8888) using your user id (e.g. s999):

`ssh -p 58022 -N -L localhost:8888:localhost:7777`
`s999@atbeetz21.informatik.tu-muenchen.de`

Now you should be able to access the notebook server from your local browser using the address `http://localhost:8888`. If your are asked for a password just hit enter.


## 7. Acknowledgments

We want to thank the **Stanford Vision Lab** and **PyTorch** for allowing us to build these exercises on material they had previously developed.
