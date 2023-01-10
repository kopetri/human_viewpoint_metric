from pathlib import Path
from learning.model import ViewQualityModel
from torch.utils.data import DataLoader
from pytorch_utils.scripts import Trainer
from learning.dataset import ViewDataset, CLASSES

if __name__ == '__main__':
    trainer = Trainer('Train view quality model')

    trainer.add_argument('--learning_rate', default=1e-04, type=float, help='Learning rate')
    trainer.add_argument('--dataset_path', required=True, type=str, help='Path to data set.')
    trainer.add_argument('--valid_path', default=None, type=str, help='Path to data set.')
    trainer.add_argument('--batch_size', default=16, type=int, help='Batch size')
    trainer.add_argument('--mlp_layers', default=[256, 64], type=int, nargs='+' ,help='Hidden layers of mlps')
    trainer.add_argument('--dropout', default=0.2, type=float, help='Dropout')
    trainer.add_argument('--filter_agreement', action='store_true', help='Whether to filter for both participants voted best for the same view.')
    trainer.add_argument('--classes', default=CLASSES, nargs='+', help='Classes to use from Modelnet40')
    trainer.add_argument('--ignore', default=[], nargs='+', help='Classes to use ignore from Modelnet40')
    trainer.add_argument('--discard_rate', default=0.0, type=float, help='How much data should be discarded.')
    trainer.add_argument('--use_triplets', default=True, type=bool, help='Use triplets or best and worst image only.')
    trainer.add_argument('--decoder', default='binary', type=str, help="Select binary or goodness.")
    trainer.add_argument('--disable_rot', action='store_true', help="Disable random rotation.")
    trainer.add_argument('--n_models', type=int, default=-1, help="Select how many models per category are used.")

    args = trainer.setup()

    if args.ignore:
        print("ignoring classes: ", args.ignore)
        args.classes = [c for c in args.classes if c not in args.ignore]
        print("using classes: ", args.classes)

    if args.disable_rot:
        print("Disabling random rotation.")

    print("Using {} decoder.".format(args.decoder))

    train_dataset = ViewDataset(path=args.dataset_path, split="train", use_rot=not args.disable_rot, use_failed=False, filter_agree=args.filter_agreement, classes=args.classes, discard_rate=args.discard_rate, use_triplets=args.use_triplets, n_instances=args.n_models)
    val_dataset   = ViewDataset(path=args.valid_path if args.valid_path else args.dataset_path, split="valid", use_rot=not args.disable_rot, use_failed=False, filter_agree=args.filter_agreement, classes=args.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.worker
    )
    
    model = ViewQualityModel(args)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
