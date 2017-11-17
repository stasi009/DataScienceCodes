# tf-bpr

BPR implemented in Tensorflow

Bayesian Personalized Ranking(BPR) is a learning algorithm for collaborative filtering first introduced in: BPR: Bayesian Personalized Ranking from Implicit Feedback. Steffen Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme, Proc. UAI 2009.   

## Notes on GPU Applicability

https://github.com/hexiangnan/theano-BPR/issues/2


## Cloud ML Modules


### Movie lens model

Runs in about an hour and a half:

```
Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0) 
max_u_id: 6040 
max_i_id: 3952 
epoch:  1 
bpr_loss:  0.718346737115 
test_loss:  0.888832 test_auc:  0.638710508152 
epoch:  2 
bpr_loss:  0.706414132327 
test_loss:  0.846649 test_auc:  0.70872795658 
epoch:  3 
bpr_loss:  0.695199506894 
test_loss:  0.807354 test_auc:  0.748564914589 
epoch:  4 
bpr_loss:  0.684583376779 
test_loss:  0.771444 test_auc:  0.7741597347 
epoch:  5 
bpr_loss:  0.67470796366 
test_loss:  0.738683 test_auc:  0.791989548821 
epoch:  6 
bpr_loss:  0.665347917446 
test_loss:  0.708624 test_auc:  0.80501552816 
epoch:  7 
bpr_loss:  0.656520227583 
test_loss:  0.681057 test_auc:  0.814839012506 
epoch:  8 
bpr_loss:  0.648171231255 
test_loss:  0.65567 test_auc:  0.822456654431 
epoch:  9 
bpr_loss:  0.640325295434 
test_loss:  0.632719 test_auc:  0.828535502284 
epoch:  10 
bpr_loss:  0.632631466958 
test_loss:  0.6113 test_auc:  0.833484377648 
```

Command to queue job:

```    
gcloud beta ml jobs submit training ml_gpu_east_6 \
--package-path=movielens \
--module-name=movielens.bpr \
--staging-bucket="gs://tf-sharknado-ml" \
--region=us-east1 \
--scale-tier=BASIC_GPU
```


### Amzn model

Runs in 6 hours. Less data than ML example but 6 times slower?

Set `staging-bucket` to your GS bucket:

```
gcloud beta ml jobs submit training amzn_gpu_east_3 \
--package-path=amzn \
--module-name=amzn.bpr \
--staging-bucket="gs://tf-sharknado-ml" \
--region=us-east1 \
--scale-tier=BASIC_GPU
```

## Cloud ML

gcloud auth login
gcloud config set project team-3-165118
deploy.sh [comment]


## BPR

Train routine takes about 10 min
.67 AUC

run w/o GPU
CUDA_VISIBLE_DEVICES="" time python bpr.py

tensorboard --logdir=logs/log_simple_stats/

## VBPR

System requirements: Needs 8 Gigs of free memory
6 hours to train
AUC .72
