# some code is from https://github.com/CCRcmcpe/scal-sdt
from typing import Any, Literal, Optional
from pathlib import Path
import warnings
import torch
import click

DType = Literal["fp16", "fp32", "bf16"]
LayerName = Literal["attn", "ff"]
StateDict = dict[str, Any]

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16
}


def save_state_dict(state: StateDict, path: str, format: Literal["pt", "safetensors"]):
    if format == "pt":
        with open(path, 'wb') as f:
            torch.save(state, f)
    elif format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ModuleNotFoundError(
                'In order to use safetensors, run "pip install safetensors"')

        state = {k: v.contiguous().to_dense() for k, v in state.items()}
        save_file(state, path)
    else:
        raise ValueError(f'Invalid format "{format}"')


def load_model(path: Path, device: str, print_ptl_info=False) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path, device=device)
    else:
        ckpt = torch.load(path, map_location=device)
        if print_ptl_info and "epoch" in ckpt and "global_step" in ckpt:
            print(
                f"[I] {path.name}: epoch {ckpt['epoch']}, step {ckpt['global_step']}")
        return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def get_layers(model: Optional[StateDict], layer_name: LayerName) -> StateDict:
    if model is not None:
        attn_k = [x for x in model.keys() if layer_name in x]
        return {l: model[l] for l in attn_k}
    else:
        return {}


def swap_dict(target: StateDict, other: StateDict) -> None:
    """ 
      Not a pure function. Modifies dict in place.
    """
    for k, v in other.items():
        if k in target:
            target[k] = v


def swap_layers(model: StateDict, other: Optional[StateDict], layer_name: LayerName) -> None:
    """ 
      Not a pure function. Modifies model in place.
    """
    if other is None:
        return
    layers = get_layers(other, layer_name)
    swap_dict(model, layers)


@click.command()
@click.option("-a", "--attention", "attn", type=click.Path(exists=True), help="Path to attention weights.")
@click.option("-f", "--feed-forward", "ff", type=click.Path(exists=True), help="Path to feed forward weights.")
@click.option("-t", "--text-encoder", "te", type=click.Path(exists=True), help="Path to text encoder weights.")
@click.option("-r", "--rest", type=click.Path(exists=True), required=True, help="Path to rest of the model weights. You must provide this.")
@click.option("-o", "--output", type=click.Path(), required=True, help="Path to output file. Must be a .safetensors or .ckpt file.")
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists.")
def main(attn: Optional[str], ff: Optional[str], te: Optional[str], rest: str, output: str, overwrite: bool):
    attn = Path(attn) if attn else None
    ff = Path(ff) if ff else None
    te = Path(te) if te else None
    rest = Path(rest)
    output = Path(output)

    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} already exists. Use --overwrite to overwrite it.")
    if not output.suffix == ".safetensors" and not output.suffix == ".ckpt":
        raise ValueError(
            f"Output file must be a `.safetensors` or `.ckpt` file. Got {output.suffix}")
    if te is None and attn is None and ff is None:
        raise ValueError(
            "Must provide either attn or te or ff. Why are you running this script?")

    unet_dtype: DType = "fp16"
    attn_model = load_model(attn, "cpu") if attn else None
    ff_model = load_model(ff, "cpu") if ff else None
    te_model = load_model(te, "cpu") if te else None
    rest_model = load_model(rest, "cpu")

    # leave TE(Maybe?) and VAE out. I don't need them.
    rest_unet_dict = {k: v.to(DTYPE_MAP[unet_dtype])
                      for k, v in rest_model.items() if k.startswith("model.diffusion_model.")}

    swap_layers(rest_unet_dict, attn_model, "attn")
    swap_layers(rest_unet_dict, ff_model, ".ff.")

    text_encoder_dict = {}
    if te_model is not None:
        text_encoder_dict = {k: v.to(DTYPE_MAP["fp32"])
                             for k, v in te_model.items() if k.startswith("cond_stage_model.transformer.")}
        if not any(text_encoder_dict.items()):
            warnings.warn(
                "No text encoder weights were found in {}.".format(te))

    output_model = {**rest_unet_dict, **text_encoder_dict}
    format = "safetensors" if output.suffix == ".safetensors" else "pt"
    save_state_dict(output_model, output, format)
    print(f"Saved to {output.absolute()}")


if __name__ == "__main__":
    main()
