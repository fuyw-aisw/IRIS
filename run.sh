echo "--MUTAG--"
nohup python main.py --dataset 'MUTAG' --num_train=50 --num_val=50 >> result_round.out & 
wait

echo "--PTC-MR--"
nohup python main.py --dataset 'PTC_MR' --num_train=90 --num_val=90 --epochs 100 >> result.out & 
wait

echo "--DHFR--"
nohup python main.py --dataset 'DHFR' --num_train=120 --num_val=120 --epochs 100 >> result_round.out & 
wait

echo "--PROTEINS--"
nohup python main.py --dataset 'PROTEINS' --num_train=300 --num_val=300 --epochs 100 >> result_round.out & 
wait

echo "--DD--"
nohup python main.py --dataset 'DD' --num_train=300 --num_val=300 --epochs 100 >> result_round.out & 
wait


echo "--AIDS--"
nohup python main.py --dataset 'AIDS' --num_train=500 --num_val=500 --epochs 100 >> result_round.out & 
wait

echo "--NCI1--"
nohup python main.py --dataset 'NCI1' --num_train=1000 --num_val=1000 --epochs 100 >> result_round.out & 
wait
