from omegaconf import OmegaConf, DictConfig


def log_hydra_config(cfg: DictConfig) -> None:
    import mlflow
    flat = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    mlflow.log_params(_flatten_dict(flat))


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v if isinstance(v, (int, float, bool)) else str(v)
    return items
