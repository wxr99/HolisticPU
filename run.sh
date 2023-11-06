############################################################# cifar10
python main.py --dataset cifar10_1 --num-labeled 1000 --arch CNN_CIFAR --batch-size 64 --lr 0.0015  --out results/pn@cifar1@1000
python main.py --dataset cifar10_2 --num-labeled 1002 --arch CNN_CIFAR --batch-size 64 --lr 0.0015 --rho 0.01  --out results/pn@cifar2@1000
###############################################################stl10
python main.py --dataset stl10_1 --num-labeled 1000 --arch CNN_STL --batch-size 64 --lr 0.001  --out results/pn@stl1@1000
python main.py --dataset stl10_2 --num-labeled 1000 --arch CNN_STL --batch-size 64 --lr 0.0015  --out results/pn@stl2@1000
###############################################################fmnist
python main.py --dataset fmnist_1 --num-labeled 1000 --arch LeNet --batch-size 64 --lr 0.002  --out results/pn@fmnist1@1000
python main.py --dataset fmnist_2 --num-labeled 1002 --arch LeNet --batch-size 64 --lr 0.002  --out results/pn@fmnist2@1000
## credit card

########################################## alzheimer
python main.py --dataset alzheimer --arch ResNet50 --batch-size 16 --lr 0.0005 --eval-step 100 --warming_steps 1000 --total-steps 2000 --out results/pn@alzheimer@769
