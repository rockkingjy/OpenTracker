c++(visual studio) implementation of Efficient Convolution Operator(ECO)tracker.

version1.0

Environment：caffe+vs2015+opencv3.x

Publication：

Details about the tracker can be found in the CVPR 2017 paper

Martin Danelljan, Goutam Bhat, Fahad Khan, Michael Felsberg. ECO Efficient Convolution Operators for Tracking. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

Please cite the above publication if you use the code or compare with the ECO tracker in your work. Bibtex entry

@InProceedings{DanelljanCVPR2017, Title = {ECO Efficient Convolution Operators for Tracking}, Author = {Danelljan, Martin and Bhat, Goutam and Shahbaz Khan, Fahad and Felsberg, Michael}, Booktitle = {CVPR}, Year = {2017} }

Contact Email flying_tan@163.com ucasws@gmail.com

installation

caffe-windows：Reference to caffe-windows installation online.

opencv : we recommend opencv3.x(opencv2.x can also be used, just replace some functions)

vs2015/vs2013 build and run: configure the third party of caffe-windows in the project property sheet.

Source code description:：

ECO.h, ECO.cpp tracker: parameter and feature parameters intialization, label generator, cos-window function, and so on.

eco_sample_update.h, eco_sample_update.cpp: Relevant functions of updating sample model (Calculating samples distance, update sample space model, etc.)

feature_extractor.h, feature_extractor.cpp: The function of feature extraction (sampling, cnn and hog feature extractor)

feature_operator.h, feature_operator.cpp: feature operator overloading, mapping function.

fftTools.h, fftTools.cpp: Complex matrix operations, fourier domain calculation.

optimize_scores.h, optimize_scores.cpp: Calculation of the score of final features map.

training.h, training.cpp: Feature training and update class

Detailed description and description follow-up update

Supplementary：

Currently caffe-windows of ECO is CPU version, the GPU version will be updated as soon as possible

The name of variables , functions of the classes are according to ECO matlab version. During debugging code, also you can reference to ECO Matlab version.

Vgg model is available at https://pan.baidu.com/s/1skVkPLN. After downloading, the path of the VGG folder in the main.cpp on the line 59,60,61 shall be modified.