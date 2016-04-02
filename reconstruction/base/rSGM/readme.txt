This is a small library implementing the ideas presented in "Large Scale Semi-Global Matching in the CPU", IV 2014

The code is delivered with a VS2013 solution and project. It compiles with 
- MSVC2013
- ICC 14.0 as part of Intel Parallel Studio XE 2013 
- GCC 4.6.3 (example makefile provided).

The demo executable only supports pgm input images of type P5. For demo imagery you can use the sequences provided as part of the 
Robust vision challenge: http://hci.iwr.uni-heidelberg.de/Static/challenge2012/. The results for the Blinking truck sequence are 
ok with the default parameter set. For other datasets/input images you might have to adapt at least the P1/P2 settings.

Performance hints
- use 64bit binaries for maximum performance (32bit is about 20% slower). 
- SGM classes should be instantiated once at the beginning and reused for every frame
  , to minimize stalls due to dynamic memory allocation. 
- multi-threading relies on OpenMP, so make sure you have it activated
- the number of stripes should be less or equal to the number of physical cores available
- performance of GCC/ICC might not be optimal, as optimization was done with MSVC

Restrictions
- Input images should have a width which is a multiple of 16. 
- The number of disparities calculated should be a multiple of 8 and not greater than 256.
- Input image depths of 8 and 16 bit are supported. 

There are two cost measures available
- classical 5x5 Census, fully optimized for speed
- Horizontally weighted 9x7 Center-Symmetric Census, where the census calculation is not fully optimized (used for Kitti Stereo Benchmark results,
  (see "Weighted Semi-Global Matching and Center-Symmetric Census Transform",
  Robert Spangenberg, Tobias Langner, Raúl Rojas, Proceedings of the "15th International Conference on Computer Analysis of Images 
  and Patterns (CAIP 2013)", York, United Kingdom.)

As SGM variants there are three possibilities available
- classical SGM (scales to 2 cores, parallelization done for cost measure and winner takes all)
- striped SGM (scales to number of stripes)
- striped SGM with a sub-sampling of 2/4 for the upper 64 disparities of 128

SGM P2 penalty adaptation is possible according to the model proposed in 
Banz, C.; Pirsch, P. & Blume, H. EVALUATION OF PENALTY FUNCTIONS FOR SEMI-GLOBAL MATCHING COST AGGREGATION ISPRS - 
International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2012, XXXIX-B3, 1-6

For license questions refer to license.txt.

History
- V1.0 initial version