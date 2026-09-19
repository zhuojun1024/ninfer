"""Keep large offline transfers from retaining whole artifacts in Linux's page cache."""

from __future__ import annotations

import os

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024

# Windows text mode rewrites newlines and treats 0x1A as end-of-file, so payload
# descriptors are always binary; elsewhere O_BINARY does not exist.
_BINARY = getattr(os, "O_BINARY", 0)

# Page-cache eviction is a Linux facility. Windows exposes no equivalent for an already-open handle, so
# the discards become no-ops there and Writeback only bounds the dirty bytes it writes.
_HAS_FADVISE = hasattr(os, "posix_fadvise")
_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else IO_CHUNK_BYTES


def open_read(path: str | os.PathLike) -> int:
    """Open an existing payload file for binary reading."""
    return os.open(path, os.O_RDONLY | _BINARY)


def read_at(fd: int, count: int, offset: int) -> bytes:
    """Read at an absolute offset. Windows has no os.pread, so seek first. Callers own their descriptor
    and do not share it between threads, so the file position is private."""
    if hasattr(os, "pread"):
        return os.pread(fd, count, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, count)


def write_at(fd: int, data: bytes | memoryview, offset: int) -> int:
    """Write at an absolute offset. Windows has no os.pwrite, so seek first (see read_at)."""
    if hasattr(os, "pwrite"):
        return os.pwrite(fd, data, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.write(fd, data)


def _flush_to_disk(fd: int) -> None:
    # fdatasync is POSIX-only; Windows flushes through fsync, which has no data-only variant.
    if hasattr(os, "fdatasync"):
        os.fdatasync(fd)
    else:
        os.fsync(fd)


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if not _HAS_FADVISE:
        return
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            _flush_to_disk(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
