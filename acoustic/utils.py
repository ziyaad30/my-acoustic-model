import torch
import torch.nn.functional as F
import matplotlib
import glob
import os

import torchaudio.transforms as transforms

matplotlib.use("Agg")
import matplotlib.pylab as plt


class Metric:
    def __init__(self):
        self.steps = 0
        self.value = 0

    def update(self, value):
        self.steps += 1
        self.value += (value - self.value) / self.steps
        return self.value

    def reset(self):
        self.steps = 0
        self.value = 0


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="slaney",
        )

    def forward(self, wav):
        padding = (1024 - 160) // 2
        wav = F.pad(wav, (padding, padding), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


def save_checkpoint(
    checkpoint_dir,
    acoustic,
    optimizer,
    step,
    loss,
    best,
    logger,
):
    state = {
        "acoustic-model": acoustic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"acoustic-training-model-{step}.pt"
    torch.save(state, checkpoint_path)
    if best:
        model_path = checkpoint_dir / "acoustic-model.pt"
        torch.save(state["acoustic-model"], model_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")
    old_model = oldest_checkpoint_path(checkpoint_dir, logger)
    if os.path.exists(old_model):
        os.remove(old_model)
        print(f"Removed {old_model}")


def load_checkpoint(
    load_path,
    acoustic,
    optimizer,
    rank,
    logger,
):
    logger.info(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    acoustic.load_state_dict(checkpoint["acoustic-model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", float("inf"))
    return step, loss

def extract_digits(f):
    digits = "".join(filter(str.isdigit, f))
    return int(digits) if digits else -1


def latest_checkpoint_path(dir_path, regex="acoustic-training-model-[0-9]*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    print(f_list)
    f_list.sort(key=lambda f: extract_digits(f))
    x = f_list[-1]
    print(f"latest_checkpoint_path:{x}")
    return x


def oldest_checkpoint_path(dir_path, logger, regex="acoustic-training-model-[0-9]*.pt", preserved=2):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: extract_digits(f))
    if len(f_list) > preserved:
        x = f_list[0]
        logger.info(f"oldest_checkpoint_path:{x}")
        return x
    return ""

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig
