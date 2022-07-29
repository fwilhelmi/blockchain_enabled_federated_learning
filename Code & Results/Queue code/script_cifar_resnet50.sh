#!/usr/bin/env bash

SIM_TIME=100000
LOGS_ENABLED=0
QUEUE_SIZE=1000
LAMBDA_ARRAY=(0.11 0.26 0.36 0.51 0.08 0.18 0.26 0.36 0.07 0.15 0.21 0.3 0.05 0.12 0.17 0.24)
BATCH_SIZE_ARRAY=(1 3 5 8 10 13 20 25 38 50 75 100 150 200) 
MU_ARRAY=(2.5)
TIMEOUT_MINING=1000
N_MINERS=10
C_P2P=25000000
SEED=12345

# compile code
cd main
./build_local
echo 'EXECUTING MULTIPLE SIMULATIONS... '
cd ..
# remove old script output file and node logs
rm output/*

# execute files
cd main
pwd
for mu in "${MU_ARRAY[@]}"
do
for lambda in "${LAMBDA_ARRAY[@]}"
do 
for batch_size in "${BATCH_SIZE_ARRAY[@]}"
do 
	echo ../output/script_output_m${mu}_l${lambda}_s${batch_size}.txt
	echo ""
	echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	echo "- EXECUTING lambda = ${lambda}, block size = ${batch_size}"
	./queue_main $SIM_TIME $LOGS_ENABLED $QUEUE_SIZE ${batch_size} ${lambda} ${mu} $TIMEOUT_MINING $N_MINERS $C_P2P $SEED ../output/script_output_m${mu}_l${lambda}_s${batch_size}.txt
	echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	echo ""
done
done
done
echo ""
echo 'SCRIPT FINISHED: OUTPUT FILE SAVED IN /output/script_output.txt'
echo ""
echo ""
