input=lbdc-1000k
batch=2048

for mode in 0 1
do
	./bench.exe ../INPUT_ML/$input 64 $mode $batch 0 0
	for threshold in 1 2 3 4
	do
		./bench.exe ../INPUT_ML/$input 64 $mode $batch 1 $threshold
	done
done

