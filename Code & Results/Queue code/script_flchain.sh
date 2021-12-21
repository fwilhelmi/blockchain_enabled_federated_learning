# define execution parameters
SIM_TIME=100000
LOGS_ENABLED=0
QUEUE_SIZE=1000

LAMBDA_ARRAY=( 0.02 0.2 2 20 )
BATCH_SIZE_ARRAY=($(seq 1 1 100)) 
MU_ARRAY=(0.01 0.05 0.1 0.2 0.3 0.5 1)
TIMEOUT_MINING=1000
N_MINERS=10
C_P2P=50000000
SEED=12345

# compile code
cd ..
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
	echo ""
	echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	echo "- EXECUTING lambda = ${lambda/,/.}, block size = ${batch_size}"
	./queue_main $SIM_TIME $LOGS_ENABLED $QUEUE_SIZE ${batch_size} ${lambda/,/.} ${mu/,/.} $TIMEOUT_MINING $N_MINERS $C_P2P $SEED ../output/script_output_m${mu/,/.}_l${lambda/,/.}_s${batch_size}.txt
	echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	echo ""
done
done
done
echo ""
echo 'SCRIPT FINISHED: OUTPUT FILE SAVED IN /output/script_output.txt'
echo ""
echo ""
