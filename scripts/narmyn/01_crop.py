"""Crop every Cityscapes leftImg8bit frame to its central 1024x1024 square.

Source frames are 2048x1024; we keep the middle square (crop box below) and
mirror the <split>/<city>/ layout under a fresh output root. Safe to re-run:
crops that already exist on disk are skipped, and each crop is written to a
temp file and atomically renamed, so a killed job never leaves a truncated
PNG that a later run would mistake for finished work.

Work is spread over a process pool (one image per task) because the job is
dominated by PNG encoding. Workers report a status back to the parent, which
owns all counting and logging so output stays ordered.

One corrupt all-black frame (troisdorf_000000_000073_leftImg8bit.png) was found
in train_extra and deleted by hand; is_valid_image() guards against any others.
"""

import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

from PIL import Image, ImageStat, UnidentifiedImageError

# ---------------------------------------------------------------- constants --
# Data and outputs live under $TRDP_ROOT when set (the pipeline working
# tree on the cluster); otherwise they resolve relative to this repo.
PROJECT_ROOT = Path(os.environ.get("TRDP_ROOT",
                                   Path(__file__).resolve().parents[2]))
INPUT_ROOT = PROJECT_ROOT / "data" / "cityscapes_dataset" / "leftImg8bit"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "cityscapes_cropped"

SPLITS = ("train", "val", "test", "train_extra")
PATTERN = "*_leftImg8bit.png"

EXPECTED_SIZE = (2048, 1024)      # (width, height) of every source frame
CROP_BOX = (512, 0, 1536, 1024)   # central 1024x1024 square
MIN_MEAN_PIXEL = 5.0              # below this the frame is considered all-black

# compress_level 1 encodes ~5x faster than Pillow's default 6 for ~13% more
# disk (34 GB -> 39 GB across the full dataset). Lossless either way.
COMPRESS_LEVEL = 1

# Beyond ~32 workers NFS bandwidth, not CPU, is the limit.
MAX_WORKERS = 32
CHUNKSIZE = 8

# Worker result statuses.
CROPPED = "cropped"
EXISTS = "exists"
INVALID = "invalid"
WRONG_SIZE = "wrong_size"
ERROR = "error"

log = logging.getLogger("crop")


def check_image(path: Path):
    """Return (is_valid, reason). reason is None when the image is usable."""
    try:
        if path.stat().st_size == 0:
            return False, "zero-byte file"
    except OSError as exc:
        return False, f"cannot stat: {exc}"

    try:
        with Image.open(path) as img:
            img.load()
            channel_means = ImageStat.Stat(img).mean
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, f"PIL cannot open: {exc}"

    mean = sum(channel_means) / len(channel_means)
    if mean < MIN_MEAN_PIXEL:
        return False, f"all-black, mean pixel {mean:.2f}"

    return True, None


def is_valid_image(path: Path) -> bool:
    """True if `path` is a non-empty, decodable, non-black image."""
    return check_image(path)[0]


def crop_one(job):
    """Crop a single frame. Returns (status, path, detail) for the parent to log.

    Runs in a worker process, so it neither logs nor raises: every outcome is
    reported back as a status.
    """
    src, dst = job

    # Resume-safe: leave crops we already produced alone, before any decode.
    if dst.exists():
        return EXISTS, src, None

    valid, reason = check_image(src)
    if not valid:
        return INVALID, src, reason

    try:
        with Image.open(src) as img:
            # Every Cityscapes frame should be 2048x1024 -- assert that, but
            # report and skip instead of crashing so one oddball can't kill
            # the run.
            if img.size != EXPECTED_SIZE:
                return WRONG_SIZE, src, f"{img.size[0]}x{img.size[1]}"

            # Write then atomically rename, so an interrupted job leaves no
            # half-written PNG for the next run to skip over.
            # format is explicit: the temp name's extension is .tmp<pid>, so
            # Pillow cannot infer PNG from it.
            tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
            try:
                img.crop(CROP_BOX).save(
                    tmp, format="PNG", compress_level=COMPRESS_LEVEL
                )
                os.replace(tmp, dst)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
    except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the pool
        return ERROR, src, f"{type(exc).__name__}: {exc}"

    return CROPPED, src, None


def iter_jobs():
    """Yield (src, dst) for every source frame, splits in fixed order."""
    for split in SPLITS:
        split_dir = INPUT_ROOT / split
        if not split_dir.is_dir():
            log.warning("split directory missing, skipping: %s", split_dir)
            continue
        for city_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for src in sorted(city_dir.glob(PATTERN)):
                yield src, OUTPUT_ROOT / split / city_dir.name / src.name


def worker_count() -> int:
    """Cores SLURM gave us, falling back to the machine's count."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return min(int(slurm), MAX_WORKERS)
    return min(os.cpu_count() or 1, MAX_WORKERS)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    if not INPUT_ROOT.is_dir():
        log.error("input root does not exist: %s", INPUT_ROOT)
        return 1

    log.info("input root:  %s", INPUT_ROOT)
    log.info("output root: %s", OUTPUT_ROOT)

    jobs = list(iter_jobs())
    if not jobs:
        log.error("no images matching %s under %s", PATTERN, INPUT_ROOT)
        return 1

    # Pre-create the city directories in the parent so workers never race.
    for _, dst in jobs:
        dst.parent.mkdir(parents=True, exist_ok=True)

    workers = worker_count()
    log.info("found %d images; cropping with %d workers", len(jobs), workers)

    counts = {CROPPED: 0, EXISTS: 0, INVALID: 0, WRONG_SIZE: 0, ERROR: 0}
    done = 0

    def record(result):
        nonlocal done
        status, src, detail = result
        counts[status] += 1
        done += 1
        if status == INVALID:
            log.warning("skip (%s): %s", detail, src)
        elif status == WRONG_SIZE:
            log.warning(
                "skip (unexpected size %s, expected %dx%d): %s",
                detail, *EXPECTED_SIZE, src,
            )
        elif status == ERROR:
            log.error("failed (%s): %s", detail, src)
        if done % 1000 == 0:
            log.info("%d/%d processed (%d cropped)", done, len(jobs), counts[CROPPED])

    if workers == 1:
        for job in jobs:
            record(crop_one(job))
    else:
        # "spawn" would re-import and re-glob in every child; fork is correct
        # here and the workers share no mutable state.
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(crop_one, jobs, chunksize=CHUNKSIZE):
                record(result)

    log.info("---------------- summary ----------------")
    log.info("total found:              %d", len(jobs))
    log.info("cropped:                  %d", counts[CROPPED])
    log.info("skipped (invalid):        %d", counts[INVALID])
    log.info("skipped (already exists): %d", counts[EXISTS])
    log.info("skipped (wrong size):     %d", counts[WRONG_SIZE])
    if counts[ERROR]:
        log.warning("failed with errors:       %d", counts[ERROR])
    log.info("output root:              %s", OUTPUT_ROOT)

    return 1 if counts[ERROR] else 0


if __name__ == "__main__":
    sys.exit(main())
