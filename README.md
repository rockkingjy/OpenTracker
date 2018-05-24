**Why most of the trackers are written by matlab? I hate that! Cpp is fast and clear! I even doubt that the FPS measured by using matlab is not really meaningful, especially for actual and embedded system use! So I will re-implement those trackers by cpp day by day, hope you like it!**


## Supported tracker:
Included                                   | Tracker    
-------------------------------------------|---------------
:ballot_box_with_check:                    | CSK          
:ballot_box_with_check:                    | KCF          
:ballot_box_with_check:                    | DSST          
:ballot_box_with_check:                    | GOTURN         

## Supported Dataset (more on progress):

Included                                   | Dataset    | Reference
-------------------------------------------|--------------|-----------
:ballot_box_with_check:                    | TLP          | [Web](https://amoudgl.github.io/tlp/)
:ballot_box_with_check:                    | UAV123       | [Web](https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx)
:ballot_box_with_check:                    | VOT-2017     | [Web](http://votchallenge.net/vot2017/dataset.html)
:ballot_box_with_check:                    | TB-2015     | [Web](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html)



## Run to compare all the trackers
```
make all
./trackerscompare.bin
```

## Run Opencv trackers
Change the path of your test images in **kcf/opencvtrackers.cpp**.
```
cd opencvtrackers
make 
./opencvtrackers.bin
```

## Run KCF / DSST
Change the path of your test images in **kcf/runkcftracker.cpp**.
```
cd kcf
make -j12
./runkcftracker.bin
```

## Run GOTURN
Change the path of your test images in **goturn/rungoturntracker.cpp**.

Need to install [[caffe](https://github.com/rockkingjy/caffe)], and change the goturn/makefile according to your installation.
### Pretrained model
You can download a pretrained [[tracker model (434 MB)](https://drive.google.com/file/d/1uc9k8sTqug_EY9kv1v_QnrDxjkrTJejR/view?usp=sharing)], put it into folder: **goturn/nets**

```
cd goturn
make -j12
./rungoturntracker.bin
```

### Run caffe classification for simple test
```
./classification.bin   /media/elab/sdd/caffe/models/bvlc_reference_caffenet/deploy.prototxt   /media/elab/sdd/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel   /media/elab/sdd/caffe/data/ilsvrc12/imagenet_mean.binaryproto   /media/elab/sdd/caffe/data/ilsvrc12/synset_words.txt   /media/elab/sdd/caffe/examples/images/cat.jpg
```
--------------------------------
## KCF Tracker

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,   
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,   
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques   
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt   
Institute of Systems and Robotics - University of Coimbra / Department of Augmented Vision DFKI   

The code is changed from [[joaofaro/KCFcpp](https://github.com/joaofaro/KCFcpp)].

## DSST Tracker
[3] M. Danelljan, G. Häger, F. Shahbaz Khan, and M. Felsberg. Discriminative Scale Space Tracking, 2016

The code is changed from [[liliumao/KCF-DSST](https://github.com/liliumao/KCF-DSST)], the max_scale_factor and min_scale_factor is set to 10 and 0.1 in case of divergence error(Especially run on UAV123 dataset when the object is quite small, ex.uav2/3/4...).

## GOTURN Tracker
**[Learning to Track at 100 FPS with Deep Regression Networks](http://davheld.github.io/GOTURN/GOTURN.html)**,
<br>
[David Held](http://davheld.github.io/),
[Sebastian Thrun](http://robots.stanford.edu/),
[Silvio Savarese](http://cvgl.stanford.edu/silvio/),
<br>
European Conference on Computer Vision (ECCV), 2016 (In press)

The code is changed from [[davheld/GOTURN](https://github.com/davheld/GOTURN)].
