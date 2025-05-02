from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def read_config(path: str | Path) -> DictConfig:
    """read a yaml file and return a dictionary

    Args:
        path (str): path to the yaml file

    Returns:
        dict: dictionary of the yaml file
    """
    # check if file exists and if is a yaml file
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a file")
    if path.suffix != ".yaml":
        raise ValueError(f"{path} is not a yaml file")
    # read the yaml file
    with open(path, "r") as file:
        d = OmegaConf.load(file)
        OmegaConf.resolve(d)

    # recursively resolve if any "config" field is referring to another config
    # file
    if "modules" in d:
        for module_name, module_info in d["modules"].items():
            if "config" in module_info and isinstance(module_info["config"], str):
                # assume the referred config path is relative to the current
                # config file
                module_info["config"] = read_config(
                    path.parent / module_info["config"]
                )

    return d
