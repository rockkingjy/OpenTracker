# Concise GOTURN

**This is a concise implementation for GOTURN: Generic Object Tracking Using Regression Networks.**

**This project is based on project  [GOTURN](http://davheld.github.io/GOTURN/GOTURN.html), I just remake GOTURN and delete some redundant files so that I can test my video Conveniently.**

**I appreciate that David Held opened his code and trained caffemodel. According to his code and help document, we can easily re-run this project and test this algorithm. But this complete project [GOTURN](http://davheld.github.io/GOTURN/GOTURN.html) have some redundant files, which will add difficults for our testers,for example folder "train","test" and "visualizer". So I implement this concise project, so that we can test this algorithm by our own video more easily.**


GOTURN appeared in this paper:

**[Learning to Track at 100 FPS with Deep Regression Networks](http://davheld.github.io/GOTURN/GOTURN.html)**,
<br>
[David Held](http://davheld.github.io/),
[Sebastian Thrun](http://robots.stanford.edu/),
[Silvio Savarese](http://cvgl.stanford.edu/silvio/),
<br>
European Conference on Computer Vision (ECCV), 2016 (In press)



## Installation

### Install dependencies:

* Install CMake:
```
sudo apt-get install cmake
```

* Install Caffe and compile using the CMake build instructions:
http://caffe.berkeleyvision.org/installation.html
You must compile caffe by cmake. If you do not use cmake to compile caffe , this project can not find required caffe.

* Install OpenCV
```
sudo apt-get install libopencv-dev
```
If you installed opencv, do not execute it.


### Compile

From the main directory, type:

open CMakeLists.txt,and change `set(Caffe_DIR your_caffe_folder)`,for example, mine is `set(Caffe_DIR ~/tracking/GOTURN/caffe)`

then
```
mkdir build
cd build
cmake ..
make
```

## Pretrained model
You can download a pretrained tracker model (434 MB) by running the following script from the main directory:

```
bash scripts/download_trained_model.sh
```

## Test your own video
```
bash scripts/runTracker.sh
```


## Run classification:
```
./classification.bin   /media/elab/sdd/caffe/models/bvlc_reference_caffenet/deploy.prototxt   /media/elab/sdd/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel   /media/elab/sdd/caffe/data/ilsvrc12/imagenet_mean.binaryproto   /media/elab/sdd/caffe/data/ilsvrc12/synset_words.txt   /media/elab/sdd/caffe/examples/images/cat.jpg
```
