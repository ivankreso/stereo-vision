all: evaluate_stereo evaluate_stereo_batch

evaluate_stereo: evaluate_stereo.cc
	g++ -std=c++11 -O3 -DNDEBUG -o evaluate_stereo evaluate_stereo.cc -lpng

evaluate_stereo_batch: evaluate_stereo_batch.cc
	g++ -std=c++11 -O3 -DNDEBUG -o evaluate_stereo_batch evaluate_stereo_batch.cc -lpng

