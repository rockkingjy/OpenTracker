# How to use the API of OpenTracker?
To use the API of the trackers is really simple, just two steps.

1. Install OpenTracker:

In folder `OpenTracker/`:
```
make -j`nproc`
sudo make install
```

2. Using the API:

Build by makefile (in folder `OpenTracker/example/`):
```
make
 ./run_eco_example.bin 
```
or build by cmake:
```
mkdir build
cmake ..
make
cd ..
./build/run_eco_example.bin 
```

## Explain the code:
Two most important API are given, take ECO for example:
```
    ecotracker.init(frame, ecobbox, parameters);
    ecotracker.update(frame, ecobbox);
```
`init` is for initialization of the tracker by the first `frame`, the bounding box `ecobbox` and the `parameters` of eco tracker given.

`update` is for update for the `frame` now, and update the result to `ecobbox`(so you can read the result from `ecobbox` directly).

Isn't that simple enough? :bird: :blush:

First, trackers should be created and initialized with grounding truth bonding box / first frame bonding box, take the example of ECO tracker:
```
    eco::ECO ecotracker;
    Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
    eco::EcoParameters parameters;
    parameters.max_score_threshhold = 0.15; // you can also change other parameters, check file: eco/parameters.hpp
    ecotracker.init(frame, ecobbox, parameters);
```
here, `ecobbox` is the bondding box for the first frame.

Then, update the tracker for each `frame`:
```
    ecotracker.update(frame, ecobbox);
```
it will update the bonding box to `ecobbox`, and that is the result.

**Attention:** If you use multi_thead for trainig ECO tracker, you need to add this at the end of the main function in case of error:
```
#ifdef USE_MULTI_THREAD
    void *status;
    int rc = pthread_join(ecotracker.thread_train_, &status);
    if (rc)
    {
        cout << "Error:unable to join," << rc << std::endl;
        exit(-1);
    }
#endif
```

For other trackers, almost the same as ECO tracker, check file `OpneTracker/example/run_opentracker_example.cc` for details.