import json
from argparse import ArgumentParser

import lightning
import numpy as np
from torch.utils.data import DataLoader

from learning.dataset import CLASSES, ViewDataset
from learning.model import ViewQualityModel
from learning.trainer import parse_ckpt

if __name__ == "__main__":
    parser = ArgumentParser("Evaluate view quality model")
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to data set.")
    parser.add_argument(
        "--filter_agreement", type=int, default=1, help="Filter for agreement between both participants."
    )
    parser.add_argument("--ckpt", type=str, default="./result/no_filter/version_5", help="Load from ckpt")
    parser.add_argument("--classes", default=CLASSES, nargs="+", help="Classes to use from Modelnet40")
    parser.add_argument("--save", type=str, default=None, help="Save result to file.")
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--precision", default="16-mixed", type=str)
    parser.add_argument("--worker", default=8, type=int)

    args = parser.parse_args()

    ckpt = parse_ckpt(args.ckpt)
    model = ViewQualityModel.load_from_checkpoint(ckpt)

    test_dataset = ViewDataset(
        path=args.dataset_path,
        split="test",
        use_failed=False,
        filter_agree=args.filter_agreement,
        classes=args.classes,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.worker)

    trainer = lightning.Trainer(
        accelerator=args.accelerator, devices=args.devices, precision=args.precision, deterministic=True
    )
    [result] = trainer.test(model=model, dataloaders=test_loader, verbose=False)

    acc = np.round(result["test_acc"], 3) * 100
    auc = np.round(result["test_auc"], 3)
    aupr = np.round(result["test_aupr"], 3)
    print("Evaluation for ckpt: \033[92m{}".format(args.ckpt) + "\033[0m")
    print("Auccuracy: {}%".format(acc))
    print("AUC: {}".format(auc))
    print("AUPR: {}".format(aupr))
    if args.save is not None:
        with open(args.save, "a") as json_file:
            json.dump(result, json_file)
