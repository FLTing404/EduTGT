#!/usr/bin/env python3
"""Download original Task 3/4 members from Eedi's ZIP using verified HTTP ranges."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
URL = "https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip"
SOURCE = "https://www.eedischool.com/projects/neurips-education-challenge"
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])))


def read_range(start, end):
    r = session.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 60), stream=True)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"Server did not return requested byte range: {r.status_code}")
    content = b"".join(r.iter_content(65536))
    if len(content) != end - start + 1:
        raise RuntimeError(f"Byte range size mismatch: {len(content)}")
    return content


def read_member_range(start, end, member):
    parts = [(a, min(a + 1024 * 1024 - 1, end)) for a in range(start, end + 1, 1024 * 1024)]
    def get_part(bounds):
        a, b = bounds
        target = ROOT / "metadata/.range_parts" / (member.split("/")[-1] + f".{a}-{b}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == b - a + 1:
            return target.read_bytes()
        block = read_range(a, b)
        target.write_bytes(block)
        print(f"Range saved {member}: {b - start + 1:,} / {end - start + 1:,}", flush=True)
        return block
    with ThreadPoolExecutor(max_workers=4) as pool:
        return b"".join(pool.map(get_part, parts))


def selected(name):
    if name.startswith("__MACOSX/") or name.endswith("/"):
        return False
    return (
        name.startswith("data/metadata/") and (name.endswith("_task_3_4.csv") or name.endswith("/subject_metadata.csv"))
        or name == "data/train_data/train_task_3_4.csv"
        or name.startswith("data/test_data/") and ("task_4" in name or "quality_response_remapped" in name)
    )


head = session.head(URL, timeout=30)
head.raise_for_status()
archive_size = int(head.headers["Content-Length"])
tail = read_range(archive_size - 65536, archive_size - 1)
eocd_at = tail.rfind(b"PK\x05\x06")
if eocd_at < 0:
    raise RuntimeError("ZIP end-of-central-directory not found")
eocd = struct.unpack_from("<4s4H2IH", tail, eocd_at)
central_size, central_offset = eocd[5], eocd[6]
central = read_range(central_offset, central_offset + central_size - 1)
entries = []
pos = 0
while central[pos:pos + 4] == b"PK\x01\x02":
    h = struct.unpack_from("<4s6H3I5H2I", central, pos)
    filename_size, extra_size, comment_size = h[10:13]
    name = central[pos + 46:pos + 46 + filename_size].decode("utf-8")
    pos += 46 + filename_size + extra_size + comment_size
    if selected(name):
        entries.append({"member": name, "local_path": "raw/" + name, "crc32": f"{h[7]:08x}", "compressed_size": h[8], "uncompressed_size": h[9], "local_header_offset": h[16], "compression_method": h[4], "source_url": URL})
if len(entries) != 10:
    raise RuntimeError(f"Expected 10 Task 3/4 CSV files, found {len(entries)}")

for entry in entries:
    member = entry["member"]
    target = ROOT / "raw" / member
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == entry["uncompressed_size"]:
        original = target.read_bytes()
        if f"{zlib.crc32(original) & 0xffffffff:08x}" == entry["crc32"]:
            entry["crc32_verified"] = True
            entry["sha256"] = hashlib.sha256(original).hexdigest()
            print(f"Verified existing {member}", flush=True)
            continue
    offset = entry["local_header_offset"]
    local = read_range(offset, offset + 29)
    header = struct.unpack("<4s5H3I2H", local)
    if header[0] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP header: {member}")
    data_start = offset + 30 + header[-2] + header[-1]
    print(f"Downloading {member}: {entry['compressed_size']:,} compressed bytes", flush=True)
    compressed = read_member_range(data_start, data_start + entry["compressed_size"] - 1, member)
    original = zlib.decompress(compressed, -15) if entry["compression_method"] == 8 else compressed
    if len(original) != entry["uncompressed_size"] or f"{zlib.crc32(original) & 0xffffffff:08x}" != entry["crc32"]:
        raise RuntimeError(f"ZIP size or CRC verification failed: {member}")
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(original)
    partial.replace(target)
    for part in (ROOT / "metadata/.range_parts").glob(member.split("/")[-1] + ".*"):
        part.unlink()
    entry["crc32_verified"] = True
    entry["sha256"] = hashlib.sha256(original).hexdigest()
    print(f"Saved {member}: {len(original):,} bytes, CRC verified", flush=True)

manifest = {
    "dataset": "Eedi / NeurIPS 2020 Education Challenge — Task 3/4 original CSV subset",
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "official_source_page": SOURCE,
    "source_archive_url": URL,
    "source_archive_bytes": archive_size,
    "source_archive_etag": head.headers.get("ETag"),
    "source_archive_last_modified": head.headers.get("Last-Modified"),
    "source_archive_content_md5_base64": head.headers.get("Content-MD5"),
    "source_archive_member_count": eocd[4],
    "download_method": "HTTP Range reads of original ZIP members; DEFLATE decoding; original member bytes preserved; ZIP CRC32 verified for every extracted member",
    "is_complete_competition_archive": False,
    "included": "All 10 published Task 3/4 training/test and related metadata CSV members, including shared subject_metadata.csv",
    "excluded": "Task 1/2 files, question images, macOS auxiliary files, starter kit",
    "csv_files_count": len(entries),
    "selected_compressed_bytes": sum(e["compressed_size"] for e in entries),
    "extracted_csv_bytes": sum(e["uncompressed_size"] for e in entries),
    "license": "CC BY-NC-ND 4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "license_source_url": SOURCE,
    "members": entries,
}
(ROOT / "metadata/download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(ROOT / "metadata/selected_zip_members.json").write_text(json.dumps({"source_url": URL, "central_directory_offset": central_offset, "central_directory_bytes": central_size, "members": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(ROOT / "metadata/SHA256SUMS").write_text("".join(f"{e['sha256']}  {e['local_path']}\n" for e in entries), encoding="utf-8")
page = session.get(SOURCE, timeout=30)
page.raise_for_status()
(ROOT / "metadata/official_source_page.html").write_bytes(page.content)
print(json.dumps({"csv_files_count": len(entries), "selected_compressed_bytes": manifest["selected_compressed_bytes"], "extracted_csv_bytes": manifest["extracted_csv_bytes"]}), flush=True)
parts_dir = ROOT / "metadata/.range_parts"
if parts_dir.exists():
    parts_dir.rmdir()
