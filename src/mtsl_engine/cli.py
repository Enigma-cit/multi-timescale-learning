import argparse
from mtsl_engine.config import EngineConfig, TrainingConfig
from mtsl_engine.training.loop import train_engine
from mtsl_engine.training.evaluation import evaluate_engine

 
def cmd_train(args: argparse.Namespace) -> None:
    cfg = EngineConfig(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    tcfg = TrainingConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
    )
    engine, metrics = train_engine(cfg, tcfg)
    print("Training finished.")
    print("Final loss:", metrics["loss"][-1])
    print("Final acc :", metrics["acc"][-1])


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = EngineConfig(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    tcfg = TrainingConfig()
    engine, _ = train_engine(cfg, tcfg)
    acc = evaluate_engine(engine, input_dim=cfg.input_dim, num_batches=args.num_batches,
                          batch_size=args.batch_size)
    print(f"Streaming eval accuracy: {acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-timescale learning engine CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train the engine on streaming data.")
    p_train.add_argument("--input_dim", type=int, default=64)
    p_train.add_argument("--hidden_dim", type=int, default=128)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--max_steps", type=int, default=5000)
    p_train.add_argument("--eval_interval", type=int, default=500)
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("eval", help="Evaluate the engine.")
    p_eval.add_argument("--input_dim", type=int, default=64)
    p_eval.add_argument("--hidden_dim", type=int, default=128)
    p_eval.add_argument("--batch_size", type=int, default=64)
    p_eval.add_argument("--num_batches", type=int, default=100)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
