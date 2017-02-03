
elements=("1000000" "10000000" "100000000")
threads=("2" "4" "8" "16")
# ./histogram "1000000" "2"

for i in ${elements[@]};
do
	for k in ${threads[@]};
	do
		./histogram $i $k
	done
done
