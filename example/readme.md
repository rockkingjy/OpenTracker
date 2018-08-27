# How to use the API of the ECO tracker?
To use the API of the trackers is really simple, just two steps.

1. Install OpenTracker/eco:
```
cd eco
make -j`nproc`
sudo make install
```

2. Using the API:
Build by makefile:
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
    ecotracker.init(frame, ecobbox);
    ecotracker.update(frame, ecobbox);
```
`init` is for initialization of the tracker by the first `frame` and the bounding box `ecobbox` given.

`update` is for update for the `frame` now, and update the result to `ecobbox`(so you can read the result from `ecobbox` directly).

Isn't that simple enough? :bird: :blush:

First, trackers should be created and initialized with grounding truth bonding box / first frame bonding box, take the example of ECO tracker:
```
    eco::ECO ecotracker;
    Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
    ecotracker.init(frame, ecobbox);
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