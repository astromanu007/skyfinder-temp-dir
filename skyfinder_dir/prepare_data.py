"""
Step 1: Download and prepare SkyFinder dataset for temperature prediction.
Fixed version - uses correct column names: Filename, CamId
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import urllib.request

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("../data/skyfinder")
IMG_DIR  = DATA_DIR / "images"
CSV_URL  = "https://cs.valdosta.edu/~rpmihail/skyfinder/analysis/complete_table_with_mcr.csv"
BASE_IMG_URL = "https://cs.valdosta.edu/~rpmihail/skyfinder/images"

# Pick 10 camera IDs (integers) — we'll auto-pick top 10 by count
MAX_IMGS_PER_CAM = 300


def download_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "complete_table.csv"
    if csv_path.exists():
        print(f"[✓] CSV already exists at {csv_path}")
        return csv_path
    print(f"[→] Downloading metadata CSV...")
    r = requests.get(CSV_URL, timeout=30)
    r.raise_for_status()
    csv_path.write_bytes(r.content)
    print(f"[✓] CSV saved to {csv_path}")
    return csv_path


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    print(f"[i] Raw CSV shape: {df.shape}")

    # Correct column names from actual CSV
    # Filename = image filename, CamId = camera ID, TempM = temp in Celsius
    temp_col = 'TempM'   # Metric temperature = Celsius
    print(f"[i] Using temp column: '{temp_col}'")

    df = df.dropna(subset=[temp_col])
    df['temperature'] = df[temp_col].astype(float)

    # Clip sensor errors
    df = df[(df['temperature'] > -40) & (df['temperature'] < 55)]

    print(f"[i] After cleaning: {len(df)} rows")
    print(f"[i] Temp range: {df['temperature'].min():.1f}°C → {df['temperature'].max():.1f}°C")
    print(f"[i] Temp mean: {df['temperature'].mean():.1f}°C, std: {df['temperature'].std():.1f}°C")

    # Set correct camera and filename columns
    df['camera']   = df['CamId'].astype(str)
    df['filename'] = df['Filename'].astype(str)

    return df


def plot_distribution(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['temperature'], bins=60, color='steelblue', edgecolor='white', linewidth=0.5)
    axes[0].set_title('Temperature Distribution (ALL data)', fontsize=13)
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(df['temperature'].mean(), color='red', linestyle='--',
                    label=f"Mean={df['temperature'].mean():.1f}°C")
    axes[0].legend()

    bins = np.arange(int(df['temperature'].min()) - 1, int(df['temperature'].max()) + 5, 5)
    counts, edges = np.histogram(df['temperature'], bins=bins)
    rare_threshold = np.percentile(counts[counts > 0], 20)
    colors = ['tomato' if c <= rare_threshold else 'steelblue' for c in counts]
    axes[1].bar(edges[:-1], counts, width=4.5, color=colors, edgecolor='white', linewidth=0.4)
    axes[1].set_title('Imbalance View (red = rare bins)', fontsize=13)
    axes[1].set_xlabel('Temperature (°C)')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Distribution plot saved → {save_path}")


def download_images(df):
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Pick top 10 cameras by image count
    top_cams = df['camera'].value_counts().head(10).index.tolist()
    print(f"[i] Top 10 cameras: {top_cams}")

    subset = df[df['camera'].isin(top_cams)].copy()
    subset = subset.groupby('camera').head(MAX_IMGS_PER_CAM).reset_index(drop=True)

    # Build URLs and local paths
    # URL pattern: BASE_IMG_URL / {CamId} / {Filename}
    subset['img_url']  = BASE_IMG_URL + "/" + subset['camera'] + "/" + subset['filename']
    subset['img_path'] = subset.apply(
        lambda r: str(IMG_DIR / r['camera'] / r['filename']), axis=1
    )

    print(f"\n[→] Attempting to download {len(subset)} images...")
    print(f"    Sample URL: {subset['img_url'].iloc[0]}")

    # Test first URL
    test_url = subset['img_url'].iloc[0]
    try:
        resp = requests.head(test_url, timeout=10)
        print(f"    URL test status: {resp.status_code}")
        if resp.status_code == 404:
            print(f"[!] 404 on test URL. The image hosting may have moved.")
            print(f"    Will try downloading anyway...")
    except Exception as e:
        print(f"[!] URL test failed: {e}")

    failed = []
    success = 0
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        out = Path(row['img_path'])
        if out.exists():
            success += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(row['img_url'], out)
            success += 1
        except Exception as e:
            failed.append((row['img_url'], str(e)))

    print(f"[i] Success: {success}, Failed: {len(failed)}")
    if failed:
        print(f"    First failure: {failed[0]}")

    return subset


def make_split_csv(subset_df):
    subset_df = subset_df[subset_df['img_path'].apply(lambda p: Path(p).exists())].copy()
    print(f"[i] Images found on disk: {len(subset_df)}")

    if len(subset_df) == 0:
        print("\n[!] NO IMAGES DOWNLOADED SUCCESSFULLY.")
        print("    The SkyFinder image server may be offline or moved.")
        print("    Creating CSV with metadata only (no images) for structure testing.")
        # Still save the metadata so you can test the pipeline structure
        subset_df = subset_df  # empty but correct columns
        # Save a note
        with open(DATA_DIR / "download_note.txt", "w") as f:
            f.write("Image download failed - server may be offline.\n")
            f.write("See README for manual download instructions.\n")

    subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(subset_df)
    train_end = int(0.7 * n)
    val_end   = int(0.85 * n)

    subset_df['split'] = 'train'
    if n > 0:
        subset_df.iloc[train_end:val_end, subset_df.columns.get_loc('split')] = 'val'
        subset_df.iloc[val_end:,         subset_df.columns.get_loc('split')] = 'test'

    out_csv = DATA_DIR / "skyfinder.csv"
    cols = [c for c in ['img_path', 'temperature', 'split', 'camera'] if c in subset_df.columns]
    subset_df[cols].to_csv(out_csv, index=False)
    print(f"[✓] Split CSV saved → {out_csv}")
    print(f"    Train: {(subset_df['split']=='train').sum()}  "
          f"Val: {(subset_df['split']=='val').sum()}  "
          f"Test: {(subset_df['split']=='test').sum()}")
    return out_csv


def check_server():
    """Check if SkyFinder image server is reachable."""
    print("\n[→] Checking SkyFinder image server...")
    test_urls = [
        "https://cs.valdosta.edu/~rpmihail/skyfinder/images/",
        "https://cs.valdosta.edu/~rpmihail/skyfinder/",
    ]
    for url in test_urls:
        try:
            r = requests.get(url, timeout=10)
            print(f"    {url} → HTTP {r.status_code}")
            if r.status_code == 200:
                print(f"    [✓] Server is up!")
                # Try to find camera folders mentioned
                if 'href' in r.text:
                    import re
                    folders = re.findall(r'href="(\d+)/"', r.text)[:10]
                    if folders:
                        print(f"    Camera folders found: {folders}")
                return True
        except Exception as e:
            print(f"    {url} → Error: {e}")
    return False


def main():
    print("=" * 60)
    print("  SkyFinder Data Preparation")
    print("=" * 60)

    csv_path = download_csv()
    df = load_and_clean(csv_path)
    plot_distribution(df, DATA_DIR / "temperature_distribution.png")

    server_ok = check_server()
    if not server_ok:
        print("\n[!] Image server appears to be down or restricted.")
        print("    Saving metadata CSV anyway for pipeline testing.")

    subset_df = download_images(df)
    make_split_csv(subset_df)

    print("\n[✓] Preparation complete!")


if __name__ == "__main__":
    main()
