from pathlib import Path


def parse_ckpt(path, return_first=True, pattern=None):
    """Resolve a checkpoint path or directory to a .ckpt file path."""
    if Path(path).is_file():
        print("Loading checkpoint: ", path)
        if return_first:
            return path
        return [path]
    ckpts = [p.as_posix() for p in Path(path).glob("**/*") if p.suffix == ".ckpt"]
    if pattern:
        ckpts = [ckpt for ckpt in ckpts if pattern in Path(ckpt).stem]
    if return_first:
        ckpt = ckpts[0]
        print("Loading checkpoint: ", ckpt)
        return ckpt
    print("Found {} checkpoints.".format(len(ckpts)))
    return ckpts
