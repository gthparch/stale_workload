input=coAuthorsDBLP
batch=2048

./bench.exe ../INPUT_ML/$input 64 $batch 0 0
for threshold in 1 2 3 4
do
	./bench.exe ../INPUT_ML/$input 64 $batch 1 $threshold
done
