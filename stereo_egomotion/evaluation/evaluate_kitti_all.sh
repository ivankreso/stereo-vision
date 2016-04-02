results_folder=$1

./evaluate_odometry_batch 0 0 1 $results_folder stats_00
./evaluate_odometry_batch 1 1 1 $results_folder stats_01
./evaluate_odometry_batch 2 2 1 $results_folder stats_02
./evaluate_odometry_batch 3 3 1 $results_folder stats_03
./evaluate_odometry_batch 4 4 1 $results_folder stats_04
./evaluate_odometry_batch 5 5 1 $results_folder stats_05
./evaluate_odometry_batch 6 6 1 $results_folder stats_06
./evaluate_odometry_batch 7 7 1 $results_folder stats_07
./evaluate_odometry_batch 8 8 1 $results_folder stats_08
./evaluate_odometry_batch 9 9 1 $results_folder stats_09
./evaluate_odometry_batch 10 10 1 $results_folder stats_10
./evaluate_odometry_batch 0 10 1 $results_folder stats_all
