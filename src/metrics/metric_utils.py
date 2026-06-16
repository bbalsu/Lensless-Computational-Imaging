from hydra.utils import instantiate


def make_metric_objects(config):
    """
    Instantiate metric objects from Hydra config.
    """
    if "metrics" not in config:
        return []

    metric_cfgs = config.metrics.get("metrics", [])
    return [instantiate(metric_cfg) for metric_cfg in metric_cfgs]
