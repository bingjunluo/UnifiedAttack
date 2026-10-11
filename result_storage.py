import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from PIL import Image


PROMPT_SAFETY_REASONS = frozenset({
    "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "MODEL_ARMOR", "JAILBREAK",
})
FINISH_SAFETY_REASONS = frozenset({
    "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT",
})


def is_safety_refusal(record):
    block = record.get("safety_block") or {}
    if not isinstance(block, dict):
        return False
    reasons = {"prompt_feedback": PROMPT_SAFETY_REASONS, "finish_reason": FINISH_SAFETY_REASONS}
    return (record.get("status") == "refused"
            and block.get("reason") in reasons.get(block.get("source"), ())
            and not record.get("text_output") and not record.get("image_paths")
            and not record.get("image_generated") and not record.get("error")
            and not record.get("image_errors"))


@contextmanager
def output_lock(directory):
    path = Path(directory) / ".run.lock"
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Output directory is in use by another process.") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Output directory is in use by another process.") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def check_run_config(args, input_bytes, pipeline_bytes):
    config = {
        "schema_version": 1,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "num_shots": args.num_shots,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "pipeline_sha256": hashlib.sha256(pipeline_bytes).hexdigest(),
        "endpoint_sha256": hashlib.sha256((args.base_url or "").encode()).hexdigest(),
    }
    directory = Path(args.output_dir)
    path = directory / "run_config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("Invalid run_config.json; choose a new --output_dir.")
        if saved != config:
            different = sorted(k for k in set(saved) | set(config) if saved.get(k) != config.get(k))
            raise ValueError("Output directory configuration differs: " + ", ".join(different)
                             + ". Choose a new --output_dir.")
        return
    log = directory / "attack_log.jsonl"
    images = directory / "generated_images"
    if ((log.exists() and log.stat().st_size)
            or (images.exists() and any(images.iterdir()))):
        raise ValueError("Existing results have no run_config.json; their configuration cannot "
                         "be verified. Choose a new --output_dir.")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def completed_ids(output_dir):
    latest = {}
    directory = Path(output_dir).resolve()
    path = directory / "attack_log.jsonl"
    if not path.exists():
        return set()
    with path.open("r+b") as handle:
        line_number = 0
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or "id" not in record:
                    raise ValueError("Expected a result object with an ID")
                if type(record["id"]) not in (int, str):
                    raise ValueError("Invalid result ID")
            except (ValueError, UnicodeDecodeError) as exc:
                if not line.endswith(b"\n") and handle.read(1) == b"":
                    handle.seek(start)
                    handle.truncate()
                    break
                raise ValueError(f"Invalid result log at line {line_number}.") from exc
            latest[record["id"]] = record
            if not line.endswith(b"\n"):
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                break
    complete = set()
    for uid, record in latest.items():
        if is_safety_refusal(record):
            complete.add(uid)
            continue
        if record.get("status") != "completed" or record.get("error") or record.get("image_errors"):
            continue
        images = record.get("image_paths", [])
        if not isinstance(images, list) or not (record.get("text_output") or images):
            continue
        if bool(record.get("image_generated")) != bool(images):
            continue
        valid = True
        for relative in images:
            try:
                if not isinstance(relative, str) or Path(relative).is_absolute():
                    raise ValueError("Expected a relative image path")
                image_path = (directory / relative).resolve()
                image_path.relative_to(directory)
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError, TypeError):
                valid = False
                break
        if valid:
            complete.add(uid)
    return complete
