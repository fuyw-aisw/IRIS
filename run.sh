for n_percents in 0.2 0.3 0.4 0.5 0.6; do
    echo "===== Running with n_percents = $n_percents ====="

    echo "--MUTAG--"
    nohup python main.py --dataset 'MUTAG' --num_train=50 --num_val=50 --epochs 100 --n_percents $n_percents >> result_n_percents.out &
    wait

    echo "--PTC-MR--"
    nohup python main.py --dataset 'PTC_MR' --num_train=90 --num_val=90 --epochs 100 --n_percents $n_percents >> result_n_percents.out &
    wait

    echo "--DHFR--"
    nohup python main.py --dataset 'DHFR' --num_train=120 --num_val=120 --epochs 100 --n_percents $n_percents >> result_n_percents.out &
    wait

    echo "--PROTEINS--"
    nohup python main.py --dataset 'PROTEINS' --num_train=300 --num_val=300 --epochs 100 --n_percents $n_percents >> result_n_percents.out &
    wait

    echo "===== Finished n_percents = $n_percents ====="
    echo ""
done