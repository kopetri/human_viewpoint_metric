from pytorch_utils import parse_ckpt
from pytorch_utils.scripts import Trainer
from learning.model import ViewQualityModel
from torch.utils.data import DataLoader
from learning.dataset import ViewDataset, CLASSES
import numpy as np
import json

if __name__ == '__main__':
    trainer = Trainer("Evaluate Cluster Separation")
    trainer.add_argument('--dataset_path', required=True, type=str, help='Path to data set.')
    trainer.add_argument('--filter_agreement', type=int, default=1, help='Whether to filter for both participants voted best for the same view.')
    trainer.add_argument('--ckpt', type=str, default="./result/no_filter/version_5", help='Load from ckpt')
    trainer.add_argument('--classes', default=CLASSES, nargs='+', help='Classes to use from Modelnet40')
    trainer.add_argument('--save', type=str, default=None, help="Save result to file.")

    args = trainer.setup(train=False)

    ckpt = parse_ckpt(args.ckpt)
    model = ViewQualityModel.load_from_checkpoint(ckpt)
    
    
    test_dataset  = ViewDataset(path=args.dataset_path, split="test",  use_failed=False, filter_agree=args.filter_agreement, classes=args.classes)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.worker
    )

    
    [result] = trainer.test(model=model, dataloaders=test_loader, verbose=False)

    acc = np.round(result["test_acc"], 3) * 100
    auc = np.round(result["test_auc"], 3)
    aupr = np.round(result["test_aupr"], 3)
    print("Evaluation for ckpt: \033[92m{}".format(args.ckpt)+'\033[0m')
    print("Auccuracy: {}%".format(acc))
    print("AUC: {}".format(auc))
    print("AUPR: {}".format(aupr))
    if not args.save is None:
        with open(args.save, "a") as json_file:
            json.dump(result, json_file)
