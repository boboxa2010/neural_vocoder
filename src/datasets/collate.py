import torch


def collate_list(items: list[dict], key: str) -> list:
    return [item[key] for item in items]


def collate_tensor(items, key: str) -> torch.Tensor:
    # suppose that all tensors have the same shape in the batch
    return torch.stack([item[key] for item in items])


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = {"text": collate_list(dataset_items, "text")}

    if "text_id" in dataset_items[0]:
        batch["text_id"] = collate_list(dataset_items, "text_id")

    if "audio_path" in dataset_items[0] and dataset_items[0]["audio_path"] is not None:
        batch["audio_path"] = collate_list(dataset_items, "audio_path")

    if "audio" in dataset_items[0] and dataset_items[0]["audio"] is not None:
        batch["audio"] = collate_tensor(dataset_items, "audio")

    if (
        "spectrogram" in dataset_items[0]
        and dataset_items[0]["spectrogram"] is not None
    ):
        batch["spectrogram"] = collate_tensor(dataset_items, "spectrogram").squeeze(1)

    return batch
