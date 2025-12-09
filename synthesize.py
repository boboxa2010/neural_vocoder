import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders, move_batch_transforms_to_device
from src.trainer import Synthesizer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Synthesizer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.synthesizer.seed)

    if config.synthesizer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.synthesizer.device

    text = config.synthesizer.get("text", None)
    text_name = config.synthesizer.get("output_name", None)

    if text is None:
        dataloaders, batch_transforms, instance_transforms = get_dataloaders(
            config, device
        )
    else:
        dataloaders, batch_transforms = None, None
        instance_transforms = instantiate(config.transforms.instance_transforms)
        move_batch_transforms_to_device(instance_transforms, device)

    model = instantiate(config.model).to(device)
    acoustic_model = instantiate(config.acoustic_model)
    print(acoustic_model)
    print(model.generator)

    metrics = instantiate(config.metrics) if "metrics" in config else None

    save_path = ROOT_PATH / "data" / "saved" / config.synthesizer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    synthesizer = Synthesizer(
        model=model,
        acoustic_model=acoustic_model,
        config=config,
        device=device,
        save_path=save_path,
        dataloaders=dataloaders,
        text=text,
        text_name=text_name,
        metrics=metrics,
        batch_transforms=batch_transforms,
        instance_transforms=instance_transforms,
        skip_model_load=False,
        resynthesize=config.synthesizer.get("resynthesize", False),
    )

    result = synthesizer.run_inference()

    if isinstance(result, dict) and all(isinstance(v, dict) for v in result.values()):
        for part in result.keys():
            for key, value in result[part].items():
                full_key = part + "_" + key
                print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
