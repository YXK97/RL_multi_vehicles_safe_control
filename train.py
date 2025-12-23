import argparse
import datetime
import os
import ipdb
import numpy as np
import yaml


def train(args):
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    if args.use_proxy:
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

    import wandb
    import jax
    import jax.numpy as jnp

    from defmarl.algo import make_algo
    from defmarl.env import make_env
    from defmarl.trainer.trainer import Trainer
    from defmarl.trainer.utils import is_connected

    n_gpu = jax.local_device_count()
    print(f"> Running train.py {args}")
    print(f"> Using {n_gpu} devices")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.debug or args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    elif not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    np.random.seed(args.seed)


    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        full_observation=args.full_observation,
        area_size=args.area_size
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        full_observation=args.full_observation,
        area_size=args.area_size
    )

     # load config
    from_iter = 0 # 预定义已训练步数
    remaining_iters = args.iters - from_iter
    model_path = None
    if args.path is not None:
        # 加载iter
        path = args.path
        model_path = os.path.join(path, "models")
        if args.from_iter is None:
            models = os.listdir(model_path)
            from_iter = max([int(model) for model in models if model.isdigit()])
        else:
            from_iter = args.from_iter

    # create algorithm
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=args.cost_weight,
        actor_gnn_layers=args.gnn_layers,
        critic_gnn_layers=args.gnn_layers,
        Vh_gnn_layers=args.Vh_gnn_layers,
        rnn_layers=args.rnn_layers,

        lr_actor=args.lr_actor,
        lr_actor_decay=args.lr_actor_decay,
        lr_actor_init=args.lr_actor_init,
        lr_actor_decay_ratio=args.lr_actor_decay_ratio,
        lr_actor_warmup_iters=args.lr_actor_warmup_iters,
        lr_actor_trans_iters=args.lr_actor_trans_iters,

        lr_critic=args.lr_critic,
        lr_critic_decay=args.lr_critic_decay,
        lr_critic_init=args.lr_critic_init,
        lr_critic_decay_ratio=args.lr_critic_decay_ratio,
        lr_critic_warmup_iters=args.lr_critic_warmup_iters,
        lr_critic_trans_iters=args.lr_critic_trans_iters,

        coef_ent=args.coef_ent,
        coef_ent_decay=args.coef_ent_decay,
        coef_ent_init=args.coef_ent_init,
        coef_ent_decay_ratio=args.coef_ent_decay_ratio,
        coef_ent_warmup_iters=args.coef_ent_warmup_iters,
        coef_ent_trans_iters=args.coef_ent_trans_iters,

        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        batch_size=args.batch_size,
        use_rnn=not args.no_rnn,
        use_lstm=args.use_lstm,
        rnn_step=args.rnn_step,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        lagr_init=args.lagr_init,
        lr_lagr=args.lr_lagr,
        iter_index=from_iter,
    )

    if model_path is not None:
        algo.load(model_path, from_iter)

    print("from_iter: ", from_iter)
    remaining_iters = args.iters - from_iter
    print("remaining_iters:", remaining_iters)

    # set up logger
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%m%d%H%M%S")
    if args.path is not None and from_iter > 0:
        log_dir=os.path.join(args.path)
    else:
        if not os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}"):
            os.makedirs(f"{args.log_dir}/{args.env}/{args.algo}", exist_ok=True)
        start_time = int(start_time)
        while os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}"):
            start_time += 1
        log_dir = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}"
    run_name = f"{args.algo}_seed{args.seed}_{start_time}"
    if args.name is not None:
        run_name = run_name + "_" + args.name

    # wandb init
    if args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader) # 这里面应当含有wandb的run_id
            run_id = config.wandb_run_id
            config = {**vars(config), **vars(args)}
            wandb.init(
                project=args.project_name,
                name=run_name,
                dir=os.path.join(args.path, "wandb", "latest-run"),
                id=run_id,
                resume_from=from_iter,
                allow_val_change=True,
                config=config,
            )
    else:
        config = {**vars(args), **(algo.config)}
        wandb.init(
            project=args.project_name,
            name=run_name,
            config=config
        )
    run_id=wandb.run.id


    # get training parameters
    train_params = {
        "run_name": run_name,
        "training_iters": args.iters,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval,
        "full_eval_interval": args.full_eval_interval,
        "start_iter": from_iter,
        "remaining_iters": remaining_iters,
    }
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        gamma=0.99,
        log_dir=log_dir,
        n_env_train_per_gpu=args.n_env_train_per_gpu,
        n_env_eval_per_gpu=args.n_env_eval_per_gpu,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
        num_gpu=n_gpu
    )

    if not args.debug:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(args, f)
            yaml.dump(algo.config, f)
            yaml.dump({"wandb_run_id": run_id}, f)
    trainer.train()

def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("-n", "--num-agents", type=int, required=True)
    parser.add_argument("--obs", type=int, required=True)
    parser.add_argument("--path", type=str, default=None)

    # algorithm arguments
    parser.add_argument("--cost-weight", type=float, default=0.)
    parser.add_argument('--lagr-init', type=float, default=0.78)
    parser.add_argument('--lr-lagr', type=float, default=1e-7)
    parser.add_argument('--clip-eps', type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-grad-norm", type=float, default=2.)

    # environment arguments
    parser.add_argument('--full-observation', action='store_true', default=False)
    parser.add_argument("--area-size", type=float, default=None)

    # training options
    parser.add_argument("--no-rnn", action="store_true", default=False)
    parser.add_argument("--n-env-train-per-gpu", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-env-eval-per-gpu", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--full-eval-interval", type=int, default=10)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--Vh-gnn-layers", type=int, default=1)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--use-lstm", action="store_true", default=False)
    parser.add_argument("--rnn-step", type=int, default=16)
    parser.add_argument("--from-iter", type=int, default=0)
    parser.add_argument("--visible-devices", type=str, default=None)
    parser.add_argument("--use-proxy", action="store_true", default=False)
    parser.add_argument("--disable-wandb", action="store_true", default=False)


    # learning rate and entropy coefficient
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-actor-decay", action="store_true", default=False)
    parser.add_argument("--lr-actor-init", type=float, default=None)
    parser.add_argument("--lr-actor-decay-ratio", type=float, default=None)
    parser.add_argument("--lr-actor-warmup-iters", type=int, default=None)
    parser.add_argument("--lr-actor-trans-iters", type=int, default=None)

    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--lr-critic-decay", action="store_true", default=False)
    parser.add_argument("--lr-critic-init", type=float, default=None)
    parser.add_argument("--lr-critic-decay-ratio", type=float, default=None)
    parser.add_argument("--lr-critic-warmup-iters", type=int, default=None)
    parser.add_argument("--lr-critic-trans-iters", type=int, default=None)

    parser.add_argument("--coef-ent", type=float, default=1e-2)
    parser.add_argument("--coef-ent-decay", action="store_true", default=False)
    parser.add_argument("--coef-ent-init", type=float, default=None)
    parser.add_argument("--coef-ent-decay-ratio", type=float, default=None)
    parser.add_argument("--coef-ent-warmup-iters", type=int, default=None)
    parser.add_argument("--coef-ent-trans-iters", type=int, default=None)

    parser.add_argument("--project-name", type=str, default="RL_vehicle_training")

    args = parser.parse_args()
    train(args)



if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
