import argparse
import os
import urllib.error
import urllib.request
import zipfile

from searchad.config.config import DATASET_DOWNLOAD_INSTRUCTIONS, SEARCHAD_HF_URL


def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(
            f"\r  Downloading... {percent:5.1f}%  ({downloaded_mb:.1f} / {total_mb:.1f} MB)",
            end="",
            flush=True,
        )
    else:
        downloaded_mb = downloaded / (1024 * 1024)
        print(f"\r  Downloading... {downloaded_mb:.1f} MB", end="", flush=True)


def _download_with_token(url: str, dest: str, hf_token: str | None) -> None:
    """Stream-download *url* to *dest*, optionally authenticating with *hf_token*."""
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 1024 * 64  # 64 KB
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    _progress_hook(1, downloaded, total_size)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise SystemExit(
                "\nHTTP 401 Unauthorized: invalid or missing HuggingFace token.\n"
                "Generate a read access token at https://huggingface.co/settings/tokens and\n"
                "pass it via --hf-token <token> or the HF_TOKEN environment variable."
            ) from e
        if e.code == 403:
            raise SystemExit(
                "\nHTTP 403 Forbidden: you must accept the dataset license before downloading.\n"
                "1. Log in to HuggingFace and visit:\n"
                "     https://huggingface.co/datasets/iis-esslingen/SearchAD\n"
                "2. Click 'Agree and access repository' to accept the license.\n"
                "3. Re-run this script with a valid --hf-token."
            ) from e
        raise


def download_dataset(
    searchad_dir: str | os.PathLike,
    hf_token: str | None = None,
) -> None:
    """
    Downloads the SearchAD annotations and default queries from HuggingFace
    and extracts them into the given output directory.

    The downloaded searchad.zip contains:
      - searchad_annotations_train.json
      - searchad_annotations_val.json
      - searchad_test_mapping_id_to_imagepath.json
      - default_queries/

    After extraction, instructions are printed for manually downloading the
    11 source dataset image archives.

    Args:
        searchad_dir: Directory where the SearchAD folder will be created.
        hf_token: HuggingFace access token for gated datasets. Falls back to
            the ``HF_TOKEN`` environment variable when *None*.
    """
    searchad_dir = os.path.abspath(searchad_dir)
    os.makedirs(searchad_dir, exist_ok=True)

    token = hf_token or os.environ.get("HF_TOKEN")

    zip_path = os.path.join(searchad_dir, "searchad.zip")

    # ── 1. Download searchad.zip ─────────────────────────────────────────────
    print("[1] Downloading SearchAD annotations from HuggingFace...")
    print()
    print("    NOTE: This is a gated dataset. To download you must:")
    print("      1. Accept the license at:")
    print("           https://huggingface.co/datasets/iis-esslingen/SearchAD")
    print("      2. Generate an access token at https://huggingface.co/settings/tokens")
    print("      3. Pass it via --hf-token <token> or set the HF_TOKEN env variable.")
    print()
    print(f"    URL:  {SEARCHAD_HF_URL}")
    print(f"    Dest: {zip_path}")
    _download_with_token(SEARCHAD_HF_URL, zip_path, token)
    print()  # newline after progress bar
    print("    Download complete.")

    # ── 2. Extract ───────────────────────────────────────────────────────────
    print(f"\n[2] Extracting {zip_path} → {searchad_dir} ...")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    # The zip contains a top-level "searchad/" prefix — strip it so the
    # contents land directly in searchad_dir instead of searchad_dir/searchad/.
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Strip the leading "searchad/" component
            rel = member.filename
            if rel.startswith("searchad/"):
                rel = rel[len("searchad/") :]
            if not rel:  # skip the top-level directory entry itself
                continue
            dest = os.path.join(searchad_dir, rel)
            if member.is_dir():
                os.makedirs(dest, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
    print("    Extraction complete.")

    # ── 3. Source dataset instructions ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("NEXT STEPS: Download the source dataset images")
    print("=" * 70)
    print(
        "SearchAD annotations reference images from 11 source datasets.\n"
        "Due to licensing restrictions, these images must be downloaded\n"
        "separately from their official hosts and placed inside:\n"
        f"  {searchad_dir}/\n"
    )
    print("Place each dataset in the corresponding subfolder shown below.\n")

    for i, ds in enumerate(DATASET_DOWNLOAD_INSTRUCTIONS, start=1):
        account_note = "  [account required]" if ds["requires_account"] else ""
        print(f"  {i:2d}. {ds['name']}{account_note}")
        print(f"      Folder : {searchad_dir}/{ds['folder']}")
        print(f"      URL    : {ds['url']}")
        print(f"      Action : {ds['instructions']}")
        print()

    print(
        "After placing all images, run the setup check to verify:\n"
        "  python searchad/check_searchad_setup.py \\\n"
        f'      --searchad-dir "{searchad_dir}"'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SearchAD annotations and default queries from HuggingFace "
        "and print instructions for downloading the source dataset images."
    )
    parser.add_argument(
        "--searchad-dir",
        type=str,
        required=True,
        help="Directory where the SearchAD folder will be created.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace access token for gated datasets. " "Can also be set via the HF_TOKEN environment variable.",
    )
    args = parser.parse_args()

    download_dataset(
        searchad_dir=args.searchad_dir,
        hf_token=args.hf_token,
    )
