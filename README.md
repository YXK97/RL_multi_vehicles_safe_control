训练指令示例：python train.py --env MVELaneChange --algo def-marl --save-interval 500 --eval-interval 100 --full-eval-interval 100 --gnn-layers 2 --Vh-gnn-layers 1 --rnn-layers 1 --num-agents 4 --obs 4 --iters
 50000 --clip-eps 0.25 --use-proxy --visible-devices 0,1 --n-env-train-per-gpu 512 --n-env-eval-per-gpu 32 --batch-size 65536 --lr-acto
r-decay --lr-actor-init 3e-4 --lr-actor-decay-ratio 30 --lr-actor-warmup-iters 20000 --lr-actor-trans-iters 10000 --lr-critic-decay --l
r-critic-init 1e-3 --lr-critic-decay-ratio 100 --lr-critic-warmup-iters 20000 --lr-critic-trans-iters 10000 --coef-ent-decay --coef-ent
-init 1e-2 --coef-ent-decay-ratio 100 --coef-ent-warmup-iters 20000 --coef-ent-trans-iters 10000



测试指令示例：def_marl_in_vehicle/test.py --epi 20 --max-step 1024 --path /home/yxk-vtd/def_marl_in_vehicle/logs/MVEDistMTarget/def-marl/seed0_1013175355
