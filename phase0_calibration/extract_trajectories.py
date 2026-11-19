import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model and tokenizer ({MODEL_NAME}) on device: {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=False
    )
    model.eval()

    input_file = Path("data/raw_calibration.json")
    if not input_file.exists():
        raise FileNotFoundError("data/raw_calibration.json not found!")

    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    trajectories = []
    human_labels = []

    print(
        f"Extracting LogitLens trajectories across 28 layers for {len(dataset)} problems...")

    for item in tqdm(dataset):
        prompt = item["prompt"]
        target_token = item["target_token"]
        human_label = item["human_category_id"]

        # Tokenize cleanly without BOS/EOS conflicts
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        target_ids = tokenizer.encode(target_token, add_special_tokens=False)

        if not target_ids:
            continue
        target_id = target_ids[0]

        layer_probs = []
        hooks = []

        # Hook function targeting Qwen2 architecture
        def make_hook():
            def hook_fn(module, args, output):
                hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]
                normed = model.model.norm(hidden_states)
                logits = model.lm_head(normed)
                prob = torch.softmax(
                    logits[0, -1, :], dim=-1)[target_id].item()
                layer_probs.append(prob)
            return hook_fn

        for layer in model.model.layers:
            hooks.append(layer.register_forward_hook(make_hook()))

        with torch.inference_mode():
            model(input_ids)

        for h in hooks:
            h.remove()

        if len(layer_probs) == len(model.model.layers):
            trajectories.append(layer_probs)
            human_labels.append(human_label)

    trajectories = np.array(trajectories)
    print(
        f"\nExtraction finished. Trajectory matrix shape: {trajectories.shape}")

    # Cluster trajectories into 8 mechanistic taxonomy buckets
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10).fit(trajectories)
    ari = adjusted_rand_score(human_labels, kmeans.labels_)

    print("\n" + "="*55)
    print(f"  CALCULATED ADJUSTED RAND INDEX (ARI): {ari:.4f}")
    print("="*55)

    if ari < 0.3:
        print("-> ARI < 0.3: Human category labels do NOT reflect model internal states.")
        print("-> ACTION: Discard human tags. Standardize on Mechanistic Centroids.")
    elif 0.3 <= ari <= 0.4:
        print("-> 0.3 <= ARI <= 0.4: Moderate alignment detected.")
        print("-> ACTION: Adopt Hybrid Taxonomy.")
    else:
        print("-> ARI > 0.4: Strong alignment detected.")
        print("-> ACTION: Retain human proxy tags.")

    # Save output for step 0.2 (DEFF calculation)
    output_data = {
        "ari": float(ari),
        "trajectories": trajectories.tolist(),
        "kmeans_labels": kmeans.labels_.tolist(),
        "human_labels": human_labels
    }
    with open("data/trajectory_results.json", "w") as f:
        json.dump(output_data, f)

    print("\nSaved trajectory matrix and cluster assignments to data/trajectory_results.json")


if __name__ == "__main__":
    main()
