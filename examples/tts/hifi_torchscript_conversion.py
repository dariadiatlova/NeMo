from nemo.collections.tts.models import HifiGanModel
from nemo.core.config import hydra_runner


@hydra_runner(config_path="conf/hifigan", config_name="hifigan")
def main(cfg):
    model_weights = "/root/storage/dasha/repos/EmoNeMo/examples/tts/nemo_experiments/HifiGan/save/checkpoint.ckpt"
    model = HifiGanModel.load_from_checkpoint(checkpoint_path=model_weights, config=cfg.model)
    model.eval()
    model.export("/root/storage/dasha/repos/EmoNeMo/examples/tts/nemo_experiments/HifiGan/save/checkpoint.pt")


if __name__ == "__main__":
    main()
