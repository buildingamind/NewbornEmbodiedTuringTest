source /home/denpak/Virts/mlagents15/bin/activate
echo $1;

run () {
	for i in {1..10}
	do
		python run.py batch_size=$1  buffer_size=$2 learning_rate=$3 cuda=$4 seed=$i
	done
			
}

case $1 in
	0)
		run 250 1000  1.0e-5 0;;
	1)
		run 500 2000  1.0e-5 1;;
	2)
		run 1000 4000 1.0e-5 2;;
	3)
		run 2000 8000 1.0e-5 3;;
	4)
		run 250 1000  1.0e-3 4;;
	5)
		run 500 2000  1.0e-3 5;;
	6)
		run 1000 4000 1.0e-3 6;;
	7)
		run 2000 8000 1.0e-3 7;;
esac


