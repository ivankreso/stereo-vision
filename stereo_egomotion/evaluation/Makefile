#all: evaluate_odometry evaluate_odometry_batch
all: evaluate_odometry_batch evaluate_odometry_tsukuba evaluate_odometry_dense evaluate_odometry_independent

evaluate_odometry: evaluate_odometry.cpp matrix.cpp
	g++ -std=c++11 -O3 -DNDEBUG -o evaluate_odometry evaluate_odometry.cpp matrix.cpp
evaluate_odometry_batch: evaluate_odometry_batch.cpp matrix.cpp
	g++ -std=c++11 -O3 -DNDEBUG -o evaluate_odometry_batch evaluate_odometry_batch.cpp matrix.cpp
evaluate_odometry_tsukuba: evaluate_odometry_tsukuba.cpp matrix.cpp
	g++ -std=c++11 -O3 -DNDEBUG -o evaluate_odometry_tsukuba evaluate_odometry_tsukuba.cpp matrix.cpp
evaluate_odometry_dense: evaluate_odometry_dense.cpp matrix.cpp
	g++ -std=c++11 -O3 -DNDEBUG -o evaluate_odometry_dense evaluate_odometry_dense.cpp matrix.cpp
evaluate_odometry_independent: evaluate_odometry_independent.cpp matrix.cpp
	g++ -std=c++11 -O3 -DNDEBUG -o evaluate_odometry_independent evaluate_odometry_independent.cpp matrix.cpp
