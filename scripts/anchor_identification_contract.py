#!/usr/bin/env python3
"""Submit and verify a DVOI access-chain head in the public Rekor log.

The project signing key is deliberately kept outside the repository.  The
contract must already pin both the project public key and Rekor's public key in
its pre-H service contract.  This program signs the canonical current-head
statement, submits a Rekor ``hashedrekord`` entry, waits for an inclusion proof,
verifies the complete receipt with the repository validator, and atomically
writes the receipt.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_identification_contract import (
    ACCESS_ANCHOR_SCHEMA,
    access_anchor_statement,
    canonical_json_bytes,
    inline_registered_artifact,
    object_hash,
    validate_access_anchor,
    validate_content_registry,
)


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    data = None if payload is None else canonical_json_bytes(payload)
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "mappo-dvoi-anchor/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def fetch_rekor_public_key(service_url: str, *, timeout: float = 30.0) -> str:
    request = urllib.request.Request(
        service_url.rstrip("/") + "/api/v1/log/publicKey",
        headers={"Accept": "application/x-pem-file", "User-Agent": "mappo-dvoi-anchor/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = response.read().decode("utf-8")
    if not value.endswith("\n"):
        value += "\n"
    if not value.startswith("-----BEGIN PUBLIC KEY-----\n"):
        raise RuntimeError("Rekor public-key endpoint did not return a PEM public key")
    return value


def generate_private_key(path: Path) -> None:
    """Create a non-overwritten P-256 project key with mode 0600."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing signing key: {path}")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        completed = subprocess.run(
            ["openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(temp_path)],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode("utf-8", "replace"))
        temp_path.chmod(0o600)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def public_key_from_private(path: Path) -> str:
    completed = subprocess.run(
        ["openssl", "pkey", "-in", str(path), "-pubout"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", "replace"))
    return completed.stdout.decode("utf-8")


def sign(path: Path, message: bytes) -> bytes:
    completed = subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", str(path)],
        input=message,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", "replace"))
    return completed.stdout


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _extract_entry(response: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(response, dict) or len(response) != 1:
        raise RuntimeError("Rekor response must contain exactly one UUID entry")
    uuid, entry = next(iter(response.items()))
    if not isinstance(uuid, str) or not isinstance(entry, dict):
        raise RuntimeError("Rekor response has an invalid UUID/entry shape")
    return uuid, entry


def submit_hashedrekord(
    service_url: str,
    body: dict[str, Any],
    *,
    timeout: float = 30.0,
) -> tuple[str, dict[str, Any]]:
    endpoint = service_url.rstrip("/") + "/api/v1/log/entries"
    try:
        response = http_json(endpoint, method="POST", payload=body, timeout=timeout)
        return _extract_entry(response)
    except urllib.error.HTTPError as error:
        # Rekor returns 409 for a byte-identical existing entry.  Its Location
        # header is authoritative; retrieving that UUID is safe and idempotent.
        if error.code != 409:
            detail = error.read().decode("utf-8", "replace")
            raise RuntimeError(f"Rekor POST failed ({error.code}): {detail}") from error
        location = error.headers.get("Location")
        if not location:
            detail = error.read().decode("utf-8", "replace")
            raise RuntimeError(f"Rekor duplicate response omitted Location: {detail}") from error
        uuid = location.rstrip("/").rsplit("/", 1)[-1]
        response = http_json(endpoint + "/" + uuid, timeout=timeout)
        found_uuid, entry = _extract_entry(response)
        if found_uuid != uuid:
            raise RuntimeError("Rekor duplicate retrieval returned a different UUID")
        return found_uuid, entry


def retrieve_with_inclusion(
    service_url: str,
    uuid: str,
    *,
    attempts: int = 20,
    interval: float = 1.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    endpoint = service_url.rstrip("/") + "/api/v1/log/entries/" + uuid
    latest: dict[str, Any] | None = None
    for attempt in range(attempts):
        found_uuid, latest = _extract_entry(http_json(endpoint, timeout=timeout))
        if found_uuid != uuid:
            raise RuntimeError("Rekor retrieval returned a different UUID")
        verification = latest.get("verification")
        if isinstance(verification, dict) and isinstance(verification.get("inclusionProof"), dict):
            return latest
        if attempt + 1 < attempts:
            time.sleep(interval)
    raise RuntimeError(f"Rekor entry {uuid} did not receive an inclusion proof")


def anchor_contract(
    contract: dict[str, Any],
    private_key: Path,
    *,
    base_dir: Path | None = None,
    attempts: int = 20,
    interval: float = 1.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    registered, content_errors = validate_content_registry(contract, base_dir)
    if content_errors:
        raise ValueError("contract content registry is invalid: " + "; ".join(content_errors))
    pre_h = contract.get("pre_h")
    if not isinstance(pre_h, dict):
        raise ValueError("contract has no pre_h record")
    service_hash = pre_h.get("anchor_service_contract_hash")
    policy_hash = pre_h.get("anchor_verification_policy_hash")
    service = inline_registered_artifact(contract, service_hash)
    if not isinstance(service, dict):
        raise ValueError("pre-H anchor service contract is not inline/content-backed")
    pinned_signer = inline_registered_artifact(
        contract, service.get("anchor_signer_public_key_hash")
    )
    actual_signer = public_key_from_private(private_key)
    if pinned_signer != actual_signer:
        raise ValueError("private key does not match the pre-H-pinned project public key")

    statement = access_anchor_statement(contract)
    statement_bytes = canonical_json_bytes(statement)
    signature = sign(private_key, statement_bytes)
    body = {
        "apiVersion": "0.0.1",
        "kind": "hashedrekord",
        "spec": {
            "data": {
                "hash": {
                    "algorithm": "sha256",
                    "value": hashlib.sha256(statement_bytes).hexdigest(),
                }
            },
            "signature": {
                "content": base64.b64encode(signature).decode("ascii"),
                "publicKey": {
                    "content": base64.b64encode(actual_signer.encode("utf-8")).decode("ascii")
                },
            },
        },
    }
    uuid, _ = submit_hashedrekord(service["service_url"], body, timeout=timeout)
    entry = retrieve_with_inclusion(
        service["service_url"],
        uuid,
        attempts=attempts,
        interval=interval,
        timeout=timeout,
    )
    receipt = {
        "schema_version": ACCESS_ANCHOR_SCHEMA,
        "anchor_service_contract_hash": service_hash,
        "anchor_verification_policy_hash": policy_hash,
        "statement": statement,
        "entry_uuid": uuid,
        "log_entry": entry,
    }
    errors = validate_access_anchor(contract, receipt, registered)
    if errors:
        raise RuntimeError("downloaded Rekor receipt failed local verification: " + "; ".join(errors))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    receipt = anchor_contract(
        contract,
        args.private_key,
        base_dir=args.contract.resolve().parent,
        attempts=args.attempts,
        interval=args.interval,
        timeout=args.timeout,
    )
    atomic_json(args.output, receipt)
    print(json.dumps({"receipt": str(args.output), "entry_uuid": receipt["entry_uuid"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
