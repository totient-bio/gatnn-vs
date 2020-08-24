from pathlib import Path
from urllib.request import urlopen
from subprocess import run
from omegaconf import OmegaConf
from tempfile import NamedTemporaryFile
from tqdm.auto import tqdm


def ensure_model(model_path=None):
    orig_cfg = OmegaConf.load(__file__ + '/../embed-config.yaml')

    if model_path is None:
        model_path = orig_cfg.model_path

    model_path = Path(model_path)
    if Path(orig_cfg.model_path).resolve() != model_path.resolve():
        return

    download_and_unpack(orig_cfg.pretrained_model_url, model_path)


def download_and_unpack(src_url, dst_path):
    if dst_path.exists():
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_filename = None
    try:
        with urlopen(src_url.pretrained_model_url) as resp:
            clen = resp.getheader('Content-length')
            if clen:
                clen = int(clen)
            print('Downloading pretrained model to ', str(dst_path))
            prog = tqdm(total=clen, unit='B', unit_scale=True)
            with NamedTemporaryFile(delete=False) as tmp:
                tmp_filename = tmp.name
                chunk = resp.read(4096)
                prog.update(len(chunk))
                while chunk:
                    tmp.write(chunk)
                    chunk = resp.read(4096)
                    prog.update(len(chunk))

        run(['tar', 'xzf', tmp_filename], cwd=dst_path.parent)

    finally:
        if tmp_filename:
            Path(tmp_filename).unlink(missing_ok=True)
