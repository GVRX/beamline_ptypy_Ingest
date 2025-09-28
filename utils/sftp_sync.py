#!/usr/bin/env python3
import argparse
import os
import stat
import sys
import time
import threading
from pathlib import Path
from queue import Queue
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

import paramiko


# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Incrementally copy a local directory to a remote server over SFTP, "
                    "uploading only files that are new or changed (size or mtime), with parallelism."
    )
    p.add_argument("source", help="Local source directory (or single file).")
    p.add_argument("dest_path", help="Remote destination directory path.")
    p.add_argument("--host", required=True, help="Remote host (hostname or IP).")
    p.add_argument("--port", type=int, default=22, help="SSH port (default: 22).")
    p.add_argument("--user", required=True, help="SSH username.")
    auth = p.add_mutually_exclusive_group()
    auth.add_argument("--password", help="SSH password.")
    auth.add_argument("--key", help="Path to private key file (PEM/OpenSSH).")
    p.add_argument("--known-hosts", default=None,
                   help="Optional path to known_hosts file (default: paramiko auto-add policy).")
    p.add_argument("--strict-host-key-checking", action="store_true",
                   help="Enable strict host key checking (fail if host key unknown).")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be transferred without actually uploading.")
    p.add_argument("--preserve-mtime", action="store_true",
                   help="Set remote file mtime to match local after upload.")
    p.add_argument("--follow-symlinks", action="store_true",
                   help="Follow local symlinks (otherwise symlinks are skipped).")
    p.add_argument("--skip-hidden", action="store_true",
                   help="Skip dotfiles and dot-directories.")
    p.add_argument("--exclude", action="append", default=[],
                   help="Exclude paths containing this substring (repeatable).")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Number of parallel SFTP workers (default: 8).")
    p.add_argument("--verbose", "-v", action="count", default=0,
                   help="Increase verbosity (repeat for more).")
    return p.parse_args()


def log(msg, *, level=1, args=None):
    if args and args.verbose >= level:
        print(msg, flush=True)


# --------------------- SSH/SFTP setup ---------------------

def host_key_policy(client, known_hosts_path, strict):
    if known_hosts_path:
        client.load_host_keys(known_hosts_path)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
    else:
        client.set_missing_host_key_policy(
            paramiko.RejectPolicy() if strict else paramiko.AutoAddPolicy()
        )


def load_pkey(key_path):
    # Try a few common key types; fall back to paramiko.AutoAddPolicy for host keys.
    loaders = (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey)
    last_exc = None
    for cls in loaders:
        try:
            return cls.from_private_key_file(key_path)
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Failed to load private key {key_path}: {last_exc}")


def connect_ssh(host, port, user, password=None, key_path=None, known_hosts=None, strict=False):
    client = paramiko.SSHClient()
    host_key_policy(client, known_hosts, strict)
    try:
        if key_path:
            pkey = load_pkey(key_path)
            client.connect(hostname=host, port=port, username=user, pkey=pkey, timeout=30)
        else:
            client.connect(hostname=host, port=port, username=user, password=password, timeout=30)
    except paramiko.SSHException as e:
        raise RuntimeError(f"SSH connection failed: {e}")
    return client


# -------------------- Connection pool --------------------

class ConnectionPool:
    """
    A simple pool of SSH/SFTP connections. Each worker borrows a pair (ssh, sftp)
    for the duration of its task, then returns it.
    """
    def __init__(self, size, connect_kwargs, args):
        self._q = Queue(maxsize=size)
        self._size = size
        self._args = args
        self._lock = threading.Lock()
        self._alive = True

        # Pre-create connections
        created = 0
        for i in range(size):
            ssh = connect_ssh(**connect_kwargs)
            sftp = ssh.open_sftp()
            self._q.put((ssh, sftp))
            created += 1
        log(f"[pool] created {created} connections", level=1, args=args)

    @contextmanager
    def borrow(self):
        pair = None
        try:
            pair = self._q.get()
            yield pair
        finally:
            if self._alive and pair is not None:
                self._q.put(pair)

    def close(self):
        with self._lock:
            self._alive = False
            # Drain and close
            while not self._q.empty():
                ssh, sftp = self._q.get_nowait()
                try:
                    sftp.close()
                except Exception:
                    pass
                try:
                    ssh.close()
                except Exception:
                    pass


# ------------------- Transfer helpers --------------------

def sftp_stat(sftp, remote_path):
    try:
        return sftp.stat(remote_path)
    except FileNotFoundError:
        return None


def should_exclude(rel_path, args):
    rel_str = str(rel_path)
    if args.skip_hidden:
        if any(p.startswith(".") for p in Path(rel_str).parts):
            return True
    for token in args.exclude:
        if token and token in rel_str:
            return True
    return False


def files_to_upload(src_root: Path, args):
    src_root = src_root.resolve()
    if src_root.is_file():
        yield Path("."), src_root.name, src_root
        return

    for root, dirs, files in os.walk(src_root, followlinks=args.follow_symlinks):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_root)

        if args.skip_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
        if args.exclude:
            dirs[:] = [d for d in dirs if not should_exclude(rel_root / d, args)]

        for name in files:
            full_path = root_path / name
            rel_path = rel_root / name
            if full_path.is_symlink() and not args.follow_symlinks:
                continue
            if should_exclude(rel_path, args):
                continue
            yield rel_root, name, full_path


def needs_transfer(sftp, local_path: Path, remote_path: str) -> bool:
    st = sftp_stat(sftp, remote_path)
    if st is None:
        return True
    try:
        lstat = local_path.stat()
    except FileNotFoundError:
        return False
    if st.st_size != lstat.st_size:
        return True
    if int(st.st_mtime) != int(lstat.st_mtime):
        return True
    return False


class DirCache:
    """
    Tracks remote directories already verified/created to avoid redundant mkdir -p.
    Thread-safe.
    """
    def __init__(self):
        self._seen = set()
        self._lock = threading.Lock()

    def mark(self, path: str):
        with self._lock:
            self._seen.add(path)

    def seen(self, path: str) -> bool:
        with self._lock:
            return path in self._seen


def ensure_remote_dir(sftp, remote_dir: str, dcache: DirCache, args):
    # Normalize to POSIX-style
    remote_dir = str(Path(remote_dir).as_posix())
    if dcache.seen(remote_dir):
        return

    # Recursively mkdir -p, with races handled
    parts = Path(remote_dir).parts
    path = ""
    for part in parts:
        path = (path + "/" + part) if path else part
        st = sftp_stat(sftp, path)
        if st is None:
            try:
                log(f"[mkdir] {path}", level=2, args=args)
                sftp.mkdir(path)
            except IOError:
                # Likely created by another thread between stat and mkdir
                st2 = sftp_stat(sftp, path)
                if st2 is None:
                    raise
                if not stat.S_ISDIR(st2.st_mode):
                    raise RuntimeError(f"Remote path exists but is not a directory: {path}")
        else:
            if not stat.S_ISDIR(st.st_mode):
                raise RuntimeError(f"Remote path exists but is not a directory: {path}")

    dcache.mark(remote_dir)


def upload_one(pool: ConnectionPool, dcache: DirCache, task, args, preserve_mtime=False, dry_run=False, verbose=False):
    rel_root, name, local_path, dest_root = task
    rel_dir = "" if str(rel_root) == "." else str(rel_root).replace("\\", "/")
    remote_path = f"{dest_root}{name}" if not rel_dir else f"{dest_root}{rel_dir}/{name}"
    remote_dir = f"{dest_root}" if not rel_dir else f"{dest_root}{rel_dir}"

    with pool.borrow() as (ssh, sftp):
        # mkdir -p (cached)
        ensure_remote_dir(sftp, remote_dir, dcache, args)

        # check if needed
        if needs_transfer(sftp, local_path, remote_path):
            if dry_run:
                print(f"[DRY] PUT {local_path} -> {remote_path}")
                return ("uploaded", local_path)
            if verbose:
                log(f"[PUT] {local_path} -> {remote_path}", level=1, args=args)
            sftp.put(str(local_path), remote_path)
            if preserve_mtime:
                lstat = local_path.stat()
                sftp.utime(remote_path, (int(lstat.st_atime), int(lstat.st_mtime)))
            return ("uploaded", local_path)
        else:
            if verbose:
                log(f"[SKIP] up-to-date: {local_path}", level=2, args=args)
            return ("skipped", local_path)


# --------------------------- Main ---------------------------

def main():
    args = parse_args()

    src = Path(args.source).expanduser()
    if not src.exists():
        print(f"Source does not exist: {src}", file=sys.stderr)
        sys.exit(2)

    dest_root = args.dest_path
    if not dest_root.endswith("/"):
        dest_root += "/"

    connect_kwargs = dict(
        host=args.host, port=args.port, user=args.user,
        password=args.password, key_path=args.key,
        known_hosts=args.known_hosts, strict=args.strict_host_key_checking
    )

    # Build pool
    try:
        pool = ConnectionPool(size=max(1, args.concurrency), connect_kwargs=connect_kwargs, args=args)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    dcache = DirCache()

    total = 0
    uploaded = 0
    skipped = 0
    errors = 0
    start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futures = []
            # Stream submission to avoid building a giant list in memory
            for rel_root, name, local_path in files_to_upload(src, args):
                total += 1
                futures.append(
                    ex.submit(
                        upload_one,
                        pool, dcache,
                        (rel_root, name, local_path, dest_root),
                        args,
                        preserve_mtime=args.preserve_mtime,
                        dry_run=args.dry_run,
                        verbose=bool(args.verbose)
                    )
                )

            for fut in as_completed(futures):
                try:
                    status, _ = fut.result()
                    if status == "uploaded":
                        uploaded += 1
                    elif status == "skipped":
                        skipped += 1
                except Exception as e:
                    errors += 1
                    print(f"[ERROR] {e}", file=sys.stderr)
    finally:
        pool.close()

    elapsed = time.time() - start
    print(f"Done. Total: {total}, uploaded: {uploaded}, skipped: {skipped}, "
          f"errors: {errors}, elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
