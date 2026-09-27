import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import importlib.util
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from configs.config import device

ADAIN_REPO_URL = "https://github.com/naoto0804/pytorch-AdaIN.git"
ADAIN_RELEASE_TAG = "v0.0.0"
ADAIN_EXPECTED_COMMIT = "324eede"

ADAIN_ROOT = Path("./pytorch-AdaIN-v0.0.0")
ADAIN_MODELS_DIR = ADAIN_ROOT / "models"
DECODER_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"

def _run_git(args, cwd=None):
    result = subprocess.run(
        ["git"] + args,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()

def download_if_missing(url, destination):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing: {destination}")
        return
    print(f"Downloading {destination.name}...")
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Failed to download {destination.name}")
    os.replace(temporary, destination)
    print(f"Downloaded {destination.name} ({destination.stat().st_size / 1024**2:.1f} MB)")

def _load_module_from_file(module_name, file_path):
    file_path = Path(file_path)
    parent_dir = str(file_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

class PretrainedAdaINModel(nn.Module):
    def __init__(self, encoder, decoder, adain_function):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adain_function = adain_function

    @torch.no_grad()
    def generate(self, content, style, alpha=1):
        content_features = self.encoder(content)
        style_features = self.encoder(style)
        transferred_features = self.adain_function.adaptive_instance_normalization(
                content_features, style_features)
        blended_features = alpha * transferred_features + (1.0 - alpha) * content_features
        output = self.decoder(blended_features)
        return output.clamp(0.0, 1.0)

def setup_adain():
    if not ADAIN_ROOT.exists():
        print("Cloning pinned AdaIN repository...")
        subprocess.run(["git", "clone", "--branch", ADAIN_RELEASE_TAG, "--depth", "1", ADAIN_REPO_URL, str(ADAIN_ROOT)], check=True)
    else:
        print(f"Using existing AdaIN repository: {ADAIN_ROOT}")

    adain_commit = _run_git(["rev-parse", "HEAD"], cwd=ADAIN_ROOT)
    if not adain_commit.startswith(ADAIN_EXPECTED_COMMIT):
        raise RuntimeError(f"Unexpected AdaIN commit.\nExpected prefix: {ADAIN_EXPECTED_COMMIT}\nFound: {adain_commit}")

    ADAIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DECODER_PATH = ADAIN_MODELS_DIR / "decoder.pth"
    VGG_PATH = ADAIN_MODELS_DIR / "vgg_normalised.pth"
    download_if_missing(DECODER_URL, DECODER_PATH)
    download_if_missing(VGG_URL, VGG_PATH)

    adain_net = _load_module_from_file("pinned_adain_net", ADAIN_ROOT / "net.py")
    adain_function = _load_module_from_file("pinned_adain_function", ADAIN_ROOT / "function.py")

    vgg = adain_net.vgg
    decoder = adain_net.decoder

    vgg.load_state_dict(torch.load(VGG_PATH, map_location="cpu", weights_only=True))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location="cpu", weights_only=True))
    
    encoder = nn.Sequential(*list(vgg.children())[:31])
    encoder.requires_grad_(False)
    decoder.requires_grad_(False)
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    trained_model = PretrainedAdaINModel(encoder, decoder, adain_function).to(device)
    trained_model.eval()
    return trained_model

@torch.no_grad()
def generate_stylized_batch(model, contents, styles, alpha=1):
    c = contents.to(device)
    s = styles.to(device)
    out = model.generate(c, s, alpha=alpha)
    return out.clamp(0, 1).cpu()
    
class CueConflictDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        p = self.pairs[i]
        return p["content_image"], p["palette_image"], p["content_class"], p["palette_class"]

def generate_all_candidates(trained_model, cue_pairs):
    imgs, cl, sl = [], [], []
    for contents, styles, c_cls, s_cls in DataLoader(CueConflictDataset(cue_pairs), batch_size=32, shuffle=False):
        imgs.append(generate_stylized_batch(trained_model, contents, styles, alpha=1.0))
        cl.append(c_cls); sl.append(s_cls)
    all_cue_cand = torch.cat(imgs)
    cand_shape = torch.cat(cl).numpy().astype(int)
    cand_tex = torch.cat(sl).numpy().astype(int)
    
    torch.save({"imgs": all_cue_cand, "shape": cand_shape, "tex": cand_tex}, "cue_conflict_candidates.pt")
    print(f"Generated {len(all_cue_cand)} candidates (saved to cue_conflict_candidates.pt)")
    return all_cue_cand, cand_shape, cand_tex