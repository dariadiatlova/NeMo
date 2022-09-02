from nemo.collections.tts.models import HifiGanModel
from nemo.core.config import hydra_runner


@hydra_runner(config_path="../conf/hifigan", config_name="hifigan")
def main(cfg):
    model_weights = "/root/storage/dasha/repos/EmoNeMo/examples/tts/nemo_experiments/HifiGan/2022-08-02_18-10-40/checkpoints/HifiGan--val_loss=0.1187-epoch=191-last.ckpt"
    model = HifiGanModel.load_from_checkpoint(checkpoint_path=model_weights, config=cfg.model)
    model.eval()
    model.export("/root/storage/dasha/saved_models/hifi_torchscript/esd16/hifi_esd_191_epoch.pt")


if __name__ == "__main__":
    main()
