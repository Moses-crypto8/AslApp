from __future__ import annotations

import argparse
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent

DEFAULT_DB = str(ROOT / "sign_language.csv")
DEFAULT_MODEL = str(ROOT / "models/asl_model.joblib")


def main():
    parser = argparse.ArgumentParser(description="Updatable ASL recognition app")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train model from the current database")
    train.add_argument("--db", default=DEFAULT_DB)
    train.add_argument("--model", default=DEFAULT_MODEL)

    recognize = sub.add_parser("recognize", help="Run webcam recognition")
    recognize.add_argument("--model", default=DEFAULT_MODEL)
    recognize.add_argument("--camera", type=int, default=0)
    recognize.add_argument("--threshold", type=float, default=0.55)

    collect = sub.add_parser("collect", help="Collect new samples for one ASL term")
    collect.add_argument("--label", required=True, help='Example: "good morning"')
    collect.add_argument("--out", required=True, help="CSV to save the new samples into")
    collect.add_argument("--camera", type=int, default=0)
    collect.add_argument("--samples", type=int, default=25)

    update = sub.add_parser("update-db", help="Append a CSV update into the main database")
    update.add_argument("--db", default=DEFAULT_DB)
    update.add_argument("--update", required=True, help="CSV containing new labeled samples")
    update.add_argument("--retrain", action="store_true", help="Retrain after updating")
    update.add_argument("--model", default=DEFAULT_MODEL)

    terms = sub.add_parser("terms", help="List terms currently in the database")
    terms.add_argument("--db", default=DEFAULT_DB)

    recognize.add_argument("--engine", choices=["normalized", "rf", "gru"], default="normalized")
    for command in (recognize, collect):
        command.add_argument("--width", type=int, help="Known original training width in pixels")
        command.add_argument("--height", type=int, help="Known original training height in pixels")
    for command in (recognize, collect):
        command.add_argument("--mode", choices=["manual", "auto"], default="manual", help="Manual SPACE boundaries recommended; auto uses motion/pause heuristics")
        command.add_argument("--max-seconds", type=float, default=6.0)
        command.add_argument("--min-seconds", type=float, default=.35)
        command.add_argument("--idle-seconds", type=float, default=.8, help="Pause length that ends an automatic recording")
        command.add_argument("--motion-threshold", type=float, default=1.2, help="Automatic motion sensitivity, palm lengths/second")
    temporal = sub.add_parser("train-temporal", help="Migrate the current CSV and train the optional GRU")
    temporal.add_argument("--db", default=DEFAULT_DB)
    temporal.add_argument("--epochs", type=int, default=12)
    sub.add_parser("gui", help="Open desktop command center")
    evaluate = sub.add_parser("evaluate", help="Evaluate normalized model on the legacy split; writes report")
    evaluate.add_argument("--db", default=DEFAULT_DB)
    evaluate.add_argument("--out", default=str(ROOT / "evaluation/reproduced.joblib"))
    args = parser.parse_args()
    if hasattr(args, "width"):
        if bool(args.width) != bool(args.height) or (args.width is not None and min(args.width,args.height) <= 0):
            parser.error("Provide both positive --width and --height, or neither.")
        reference_size = (args.width, args.height) if args.width else None
    if args.command == "train-temporal":
        if args.epochs < 1:
            parser.error("Epochs must be positive.")
        from v2.migrate import migrate
        from v2.train_v2 import run
        migrate(args.db, ROOT / "v2_data")
        result = run(ROOT / "v2_data/legacy_features_v2.npz", ROOT / "v2_runs/gru_baseline", epochs=args.epochs)
        print(f"GRU diagnostic validation: {result['final_checkpoint_accuracy']:.2%}")
        return
    if args.command == "gui":
        from desktop import launch
        launch()
        return
    if args.command == "evaluate":
        from trainer import train_model
        report = train_model(args.db, args.out)
        print(f"Reproduced holdout: {report['accuracy']:.2%} ({report['correct']}/{report['test_samples']})")
        print(report['warning'])
        return

    if args.command == "train":
        from trainer import train_model
        result = train_model(args.db, args.model)
        print(f"Trained {result['terms']} terms from {result['rows']} rows.")
        print(f"Validation accuracy: {result['accuracy']:.2%}")
        print(f"Saved model: {result['model_path']}")

    elif args.command == "recognize":
        from recognizer import run_recognizer
        model = args.model
        if args.engine == "gru" and model == DEFAULT_MODEL:
            model = str(ROOT / "v2_runs/gru_baseline/best_gru.pt")
        run_recognizer(model, args.camera, args.threshold, args.engine, reference_size, args.max_seconds, args.min_seconds, args.mode, args.idle_seconds, args.motion_threshold)

    elif args.command == "collect":
        from collector import collect_samples
        collect_samples(args.label, args.out, args.camera, args.samples, reference_size, args.max_seconds, args.min_seconds, args.mode, args.idle_seconds, args.motion_threshold)

    elif args.command == "update-db":
        from database import append_update
        from trainer import train_model
        result = append_update(args.db, args.update)
        print(f"Added {result['added_rows']} rows.")
        print(f"Labels added: {', '.join(result['labels_added'])}")
        print(f"Total rows: {result['total_rows']}")
        if result["backup"]:
            print(f"Backup created: {result['backup']}")

        if args.retrain:
            training = train_model(args.db, args.model)
            print(f"Retrained model. Accuracy: {training['accuracy']:.2%}")

    elif args.command == "terms":
        from database import list_terms
        counts = list_terms(args.db)
        print(counts.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        raise SystemExit(f"Error: {exc}")