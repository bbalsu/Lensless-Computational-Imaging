import torch


def collate_fn(dataset_items: list[dict]):
    """
    Convert list of dataset samples into one batch.

    Input:
        [
            {"measurement": Tensor, "target": Tensor, ...},
            {"measurement": Tensor, "target": Tensor, ...},
        ]

    Output:
        {
            "measurement": Tensor[B, C, H, W],
            "target": Tensor[B, C, H, W],
            ...
        }
    """
    result_batch = {}

    keys = dataset_items[0].keys()

    for key in keys:
        values = [item[key] for item in dataset_items]

        if torch.is_tensor(values[0]):
            result_batch[key] = torch.stack(values, dim=0)
        else:
            result_batch[key] = values

    return result_batch
