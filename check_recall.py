"""Check which retrieved chunks pass the 5% IoU-overlap threshold against a
ground truth source range.

Usage: run the script, then follow the prompts:
  - Enter the ground truth range as: start,end
  - Enter each retrieved chunk range as: start,end
  - Enter a blank line when you're done entering chunks
"""

from __future__ import annotations

OVERLAP_THRESHOLD: float = 0.05


def parse_range(raw: str) -> tuple[int, int]:
    """Parse a 'start,end' string into a (start, end) tuple of ints."""
    parts: list[str] = raw.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 'start,end', got: {raw!r}")
    start: int = int(parts[0])
    end: int = int(parts[1])
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")
    return start, end


def compute_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Compute intersection-over-union between two [start, end) ranges."""
    a_start, a_end = a
    b_start, b_end = b

    intersection: int = max(0, min(a_end, b_end) - max(a_start, b_start))

    union: int = (a_end - a_start) + (b_end - b_start) - intersection

    if union == 0:
        return 0.0

    return intersection / union


def main() -> None:
    gt_raw: str = input("Ground truth range (start,end): ").strip()
    gt_range: tuple[int, int] = parse_range(gt_raw)

    chunk_ranges: list[tuple[int, int]] = []
    print("Enter chunk ranges one per line (blank line to finish):")

    while True:
        raw: str = input(f"  Chunk {len(chunk_ranges) + 1}: ").strip()
        if not raw:
            break
        chunk_ranges.append(parse_range(raw))

    if not chunk_ranges:
        print("No chunks entered.")
        return

    print()
    print(f"Ground truth: {gt_range[0]}-{gt_range[1]} (length {gt_range[1] - gt_range[0]})")
    print(f"Threshold: {OVERLAP_THRESHOLD:.0%}")
    print("=" * 50)

    any_pass: bool = False

    for i, chunk_range in enumerate(chunk_ranges, start=1):
        iou: float = compute_iou(chunk_range, gt_range)
        passed: bool = iou >= OVERLAP_THRESHOLD
        any_pass = any_pass or passed

        status: str = "PASS" if passed else "fail"
        length: int = chunk_range[1] - chunk_range[0]

        print(
            f"Chunk {i} [{chunk_range[0]}-{chunk_range[1]}] "
            f"(len {length}): IoU = {iou:.4f} ({iou:.2%}) -> {status}"
        )

    print("=" * 50)
    print(f"Source found (recall counts this GT as retrieved): {any_pass}")


if __name__ == "__main__":
    main()
