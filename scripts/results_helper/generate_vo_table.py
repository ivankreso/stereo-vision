#!/usr/bin/python

import sys

ordering_fname = sys.argv[1]
results_fname = sys.argv[1]
exp_order = open(ordering_fname).read().split()

results = open(results_fname).read().split()

out_file = open("table.txt")

for exp_name in exp_order:
    for i in range(0, len(results), 3):
        if exp_name == results[i]:
            print(exp_name)
            print(results[i])
            print(results[i+1])
            print(results[i+2])


