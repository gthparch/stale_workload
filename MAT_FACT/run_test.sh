input=news20.binary
batch=2048

for mode in 0
do
	./bench.exe $input 19996 1355191 3 200 64 $mode $batch 0 0 
	for threshold in 1 2 3 4
	do
		./bench.exe $input 19996 1355191 3 200 64 $mode $batch 1 $threshold 
	done
done


