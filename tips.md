
## Parameter analysis:

In eco/params.h/search_area_scale = 4.5, if change to 2.5, ECO will lose the target in UAV123/person23.

While, in kcf/kcftracker.cpp/padding = 2.5, if change to 4.5, KCF will follow the target in the same
data above.

In eco/params.h/nSamples = 50, if change to 10, ECO will lose the target in VOT/girl.

But, in eco/params.h/nSamples = 50, if change to 10, ECO will keep the target in UAV123/bike1.

The reason is for VOT/girl, it needs more history memory to return back to target. But for UAV123/bike1, long histroy means cannot adapte tothe recent changes. There is a dilemma in here.


## Running time analysis:

For the localization, eco/feature_extractor.cpp/get_hog() occupies most of the time, so diminute the 
eco/params.h/number_of_scales will speed-up the algorithm. For gpu vesion, should try to rewrite this function.

For the training, if just sampling update, the time it takes is quite few. But if training (every eco/params.h/tarin_gap time), it takes lot of time. 


## Dataset testing:

Dataset that ECO_HOG get lost, KCF keeps correct:
**UAV123/wakeboard1** 

Dataset that ECO_HOG get lost, GOTURUN keeps correct:
**VOT/iceskater1** 
**TLP/IceSkating** 

Dataset that ECO_HOG get lost, ALL the others keeps correct:
**TLP/Drone3** 

Just ECO_HOG right:
**TLP/Sam**

No one right:
**VOT/glove**

## Other tips:

VOT dataset gives the bounding box as float, it can be used directly for ECO and GOTURN, the others needs to be converted to int.

For timer `double timercv = (double)getTickCount();`, it should be double, float is not enough.

For `frame.copyTo(frameDraw);`, onley this way, it copy the memory, while `frameDraw = frame` does not.
