"""Import a folder of images/videos and their captions as a UI dataset.

    python scripts/import_dataset.py <name> <dir>

Copies every media file under <dir> (subfolders kept) plus its caption
(<stem>.txt next to it) into <DATASETS_FOLDER>/<name>, which is all the UI
needs to list it on the Datasets page. The name is normalized the same way
the UI's "New Dataset" dialog does it (lowercase, runs of other characters ->
'_'). Refuses to touch an existing dataset.
"""
import argparse
import os
import re
import shutil
import sys

import aitk_ui

# same list the UI's datasets/listImages route accepts
MEDIA_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp",
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".flv",
    ".mp3", ".wav", ".flac", ".ogg",
}


def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower())


def find_media(src):
    for root, dirs, files in os.walk(src):
        # skip .thumbs, _latent_cache and other toolkit/hidden sidecar dirs
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
        for f in sorted(files):
            if not f.startswith(".") and os.path.splitext(f)[1].lower() in MEDIA_EXTS:
                yield os.path.relpath(os.path.join(root, f), src)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help="dataset name as shown in the UI")
    parser.add_argument("dir", help="folder with media files and <stem>.txt captions")
    parser.add_argument("--caption-ext", default="txt", help="caption extension (default: txt)")
    parser.add_argument("-n", "--dry-run", action="store_true", help="report what would be copied")
    args = parser.parse_args()

    src = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(src):
        sys.exit(f"error: {src} is not a directory")

    name = normalize_name(args.name)
    if not name.strip("_"):
        sys.exit(f"error: invalid dataset name {args.name!r}")
    if name != args.name:
        print(f"dataset name normalized: {args.name!r} -> {name!r}")

    try:
        settings = aitk_ui.get_settings()
        ui_up = True
    except aitk_ui.UIError as e:
        print(f"warning: {e}; reading DATASETS_FOLDER from the database instead")
        settings = aitk_ui.get_settings_offline()
        ui_up = False
    root = os.path.abspath(settings["DATASETS_FOLDER"])
    dest = os.path.join(root, name)
    if os.path.commonpath([src, dest]) in (src, dest):
        sys.exit(f"error: source {src} and destination {dest} overlap")
    if os.path.exists(dest):
        sys.exit(f"error: dataset {name!r} already exists at {dest}")

    media = list(find_media(src))
    if not media:
        sys.exit(f"error: no media files found in {src}")
    pairs, uncaptioned = [], []
    for rel in media:
        cap = os.path.splitext(rel)[0] + "." + args.caption_ext
        if os.path.isfile(os.path.join(src, cap)):
            pairs.append((rel, cap))
        else:
            pairs.append((rel, None))
            uncaptioned.append(rel)

    print(f"{len(media)} media files, {len(media) - len(uncaptioned)} with captions -> {dest}")
    if uncaptioned:
        shown = ", ".join(uncaptioned[:5]) + (" ..." if len(uncaptioned) > 5 else "")
        print(f"warning: {len(uncaptioned)} without a .{args.caption_ext} caption: {shown}")
    if args.dry_run:
        return

    # build under a hidden temp name so a failed copy never shows up in the UI
    tmp = os.path.join(root, f".{name}.importing")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        for rel, cap in pairs:
            for f in (rel, cap):
                if f is None:
                    continue
                os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True)
                shutil.copy2(os.path.join(src, f), os.path.join(tmp, f))
        os.rename(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    if ui_up:
        listed = aitk_ui.request("POST", "/api/datasets/listImages", {"datasetName": name})
        print(f"imported: UI lists {len(listed['images'])} items in dataset {name!r}")
    else:
        print(f"imported to {dest} (UI not running; it will show up once it is)")


if __name__ == "__main__":
    main()
