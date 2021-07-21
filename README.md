<div align="center">

<img src="imgs/doge-logo.png">

**The image classifier for dog breeds built with PyTorch on Tsinghua Dogs 
Dataset [[1]](#1).**

---

</div>

Doggofier is a web application that allows an user to upload JPEG photo of a dog and CNN running in the background would predict a breed of the dog with certain confidence. As a main deep learning technology PyTorch framework was used with its supporting library Torchvision for computer vision tasks. For experiments two popular architectures were chosen: ResNet-50 [[2]](#2) and VGG-16 [[3]](#3), both pretrained on the ImageNet dataset and fine-tuned on the Tsinghua Dogs dataset with customized classifiers. Also the experimentation interface was prepared so it's possible to easily add a new model and train it from the command line with hyperparameters specified in JSON file. The application itself was developed with the usage of Flask framework and contenerized in Docker to isolate the whole environment. Eventually deployment to Heroku was performed with Gunicorn as a WSGI server.

To find more information about the project head over to the [application website](https://doggofier.herokuapp.com/).

## Installation

First acquire the source code by cloning the git repository, then install all dependencies and finally run `setup.py`:

```
git clone https://github.com/mickuz/doggofier.git

pip install -r requirements.txt

python setup.py install
```

## Getting started

If you want to create and train your own models make sure to download the dataset by executing `download_data` script. Depending on the operating system you may need to change access permissions for the file. You can add new architectures in `doggofier/models.py` file by inheritance of general `Model` class. To train a model run the following command:

```
doggofier/train.py [--data_dir] [--model_dir] [--cat_file] params_file model
```

Similarly you can run the script evaluating the model:

```
doggofier/evaluate.py [--data_dir] [--model_dir] [--cat_file] params_file model
```

If you want to contenerize and deploy the application to Heroku you need to have Docker installed and you must be signed up [here](https://www.heroku.com/). First build an image of your container:

```
docker image build -t doggofier-app .
```

Next you need to login to Heroku CLI:

```
heroku container:login
```

This command will open the browser and prompt you to pass your Heroku credentials. You should receive a message "Login Succeeded" if everything was fine. To create an application run the command below.

```
heroku create <app-name>
```

It should create a link to the application. Finally you need to push and release the container into Heroku by executing the following two commands:

```
heroku container:push web --app <app-name>

heroku container:release web --app <app-name>
```

At this point the application should be released and should be running on Heroku in the site created earlier.

## References

<a id="1">[1]</a> 
Zou, Ding-Nan and Zhang, Song-Hai and Mu, Tai-Jiang and Zhang, Min (2020). 
A new dataset of dog breed images and a benchmark for fine-grained
classification.
*Computational Visual Media 6, 477-487.*

<a id="2">[2]</a> 
He, Keiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian (2016). 
Deep Residual Learning for Image Recognition.
*2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.*

<a id="3">[3]</a> 
Simonyan, Karen and Zisserman, Andrew (2015). 
Very Deep Convolutional Networks for Large-Scale Image Recognition.
*Computing Research Repository, abs/1409.1556.*
