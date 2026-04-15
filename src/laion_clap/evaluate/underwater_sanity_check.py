import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from laion_clap import CLAP_Module


def load_labels(label_path: Path):
    with label_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError("Label file must be a JSON object: {label_name: index}.")
    return [k for k, _ in sorted(mapping.items(), key=lambda x: x[1])]


def main():
    parser = argparse.ArgumentParser(
        description="Run a quick underwater audio-text similarity sanity check with CLAP."
    )
    parser.add_argument("--audio-file", type=str, required=True, help="Path to one underwater audio file.")
    parser.add_argument("--class-label-path", type=str, required=True, help="Path to class label JSON.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional local CLAP checkpoint path.")
    parser.add_argument("--amodel", type=str, default="HTSAT-tiny")
    parser.add_argument("--tmodel", type=str, default="roberta")
    parser.add_argument("--enable-fusion", action="store_true", default=False)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="This is an underwater sound of {}.",
        help="Prompt template where {} will be replaced by class name.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    label_path = Path(args.class_label_path).resolve()
    audio_path = Path(args.audio_file).resolve()
    if not label_path.exists():
        raise FileNotFoundError(f"class-label-path not found: {label_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"audio-file not found: {audio_path}")

    model = CLAP_Module(
        enable_fusion=args.enable_fusion,
        amodel=args.amodel,
        tmodel=args.tmodel,
    )
    if args.ckpt:
        model.load_ckpt(ckpt=str(Path(args.ckpt).resolve()))
    else:
        model.load_ckpt()

    class_names = load_labels(label_path)
    prompts = [args.prompt_template.format(c) for c in class_names]

    with torch.no_grad():
        audio_embed = torch.from_numpy(model.get_audio_embedding_from_filelist([str(audio_path)]))
        text_embed = torch.from_numpy(model.get_text_embedding(prompts))
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        scores = (audio_embed @ text_embed.T).squeeze(0)
        top_k = max(1, min(args.top_k, len(class_names)))
        values, indices = torch.topk(scores, k=top_k)

    print(f"audio_file: {audio_path}")
    print(f"labels: {len(class_names)}")
    print("top predictions:")
    for rank, (idx, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        print(f"{rank:>2}. {class_names[idx]}  score={score:.6f}")


if __name__ == "__main__":
    main()
