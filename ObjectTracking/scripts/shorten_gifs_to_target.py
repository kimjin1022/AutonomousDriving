# shorten_gifs_to_target.py
# 사용법:
#   python shorten_gifs_to_target.py --dir ObjectTracking/outputs/vis --target-mb 20
#   python shorten_gifs_to_target.py --dir ObjectTracking/outputs/vis --target-mb 20 --overwrite
#
# 동작:
# - 폴더 내 *.gif 전부에 대해 현재 용량을 확인
# - 크면 프레임 수를 점점 줄여(영상 길이 단축) 다시 저장
# - 목표 용량 이하가 되면 종료
# - 기본은 *_short.gif로 저장, --overwrite로 원본 교체 가능

import argparse
import os
from PIL import Image, ImageSequence

def human(bytes_):
    return f"{bytes_ / (1024*1024):.2f} MB"

def count_frames_and_median_duration(img: Image.Image, default_ms=100):
    # 프레임 개수와 duration(중앙값)을 추정
    durations = []
    n = 0
    for f in ImageSequence.Iterator(img):
        n += 1
        d = f.info.get("duration", 0)
        if d > 0:
            durations.append(int(d))
    if durations:
        durations.sort()
        dur = durations[len(durations)//2]
    else:
        dur = default_ms
    return n, max(1, int(dur))

def save_first_k_frames(in_path, out_path, k, duration_ms, optimize=True):
    """앞에서 k프레임만 저장. duration은 단일 정수로 줘서 PIL 오류 방지."""
    with Image.open(in_path) as im:
        it = ImageSequence.Iterator(im)
        frames = []
        for i, f in enumerate(it):
            if i >= k:
                break
            # 그대로 저장해도 PIL이 자동 팔레트 변환해줌
            frames.append(f.copy())
        if not frames:
            raise RuntimeError("No frames to save.")

        # 저장 (duration을 단일값으로 전달 → 길이 mismatch 에러 방지)
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration_ms,
            optimize=optimize,
            disposal=2,  # 이전 프레임 폐기(대개 용량↓)
        )

def shrink_gif_by_length(in_path, out_path, target_bytes,
                         min_keep=1, step_ratio=0.85, max_iters=20, verbose=True):
    """프레임 수를 점점 줄이며 target 이하가 될 때까지 저장."""
    with Image.open(in_path) as im0:
        total_frames, duration_ms = count_frames_and_median_duration(im0)

    # 처음엔 전체 프레임으로 저장해 보고, 크면 줄이기 시작
    keep = total_frames
    last_size = None
    for it in range(max_iters):
        save_first_k_frames(in_path, out_path, keep, duration_ms, optimize=True)
        sz = os.path.getsize(out_path)
        if verbose:
            print(f"  try#{it+1}: keep={keep}/{total_frames} frames -> {human(sz)}")

        if sz <= target_bytes:
            return True, sz, (keep, total_frames, duration_ms)

        # 더 줄이기
        if keep <= min_keep:
            last_size = sz
            break
        new_keep = max(min_keep, int(keep * step_ratio))
        if new_keep == keep and keep > min_keep:
            new_keep = keep - 1
        keep = new_keep
        last_size = sz

    return False, last_size, (keep, total_frames, duration_ms)

def main():
    ap = argparse.ArgumentParser(description="Shorten GIFs (by cutting length) until under target size.")
    ap.add_argument("--dir", type=str, default="ObjectTracking/outputs/vis", help="GIF 폴더 경로")
    ap.add_argument("--target-mb", type=float, default=20.0, help="목표 용량(MB)")
    ap.add_argument("--overwrite", action="store_true", help="원본 교체")
    ap.add_argument("--step-ratio", type=float, default=0.85, help="프레임 감축 비율(0~1, 작을수록 빨리 줄임)")
    ap.add_argument("--max-iters", type=int, default=20, help="최대 반복 저장 횟수")
    args = ap.parse_args()

    target_bytes = int(args.target_mb * 1024 * 1024)

    if not os.path.isdir(args.dir):
        print(f"[ERR] Not a directory: {args.dir}")
        return

    gifs = [f for f in os.listdir(args.dir) if f.lower().endswith(".gif")]
    if not gifs:
        print("[INFO] No GIF files found.")
        return

    print(f"[INFO] Target: {args.target_mb:.1f} MB | Dir: {args.dir}")
    for fn in gifs:
        src = os.path.join(args.dir, fn)
        dst = src if args.overwrite else os.path.join(args.dir, f"{os.path.splitext(fn)[0]}_short.gif")

        orig_sz = os.path.getsize(src)
        print(f"- {fn}: {human(orig_sz)}")

        if orig_sz <= target_bytes and not args.overwrite:
            print("  already under target. skip (kept original).")
            continue

        ok, final_sz, info = shrink_gif_by_length(
            src, dst, target_bytes,
            step_ratio=args.step_ratio,
            max_iters=args.max_iters,
            verbose=True,
        )
        keep, total, dur = info
        if ok:
            print(f"  ✅ done: {human(final_sz)} (kept {keep}/{total} frames, duration={dur}ms) -> {os.path.basename(dst)}")
        else:
            print(f"  ⚠️  could not reach target (best {human(final_sz)} with {keep}/{total} frames). Left the shortest version.")

    print("[DONE]")

if __name__ == "__main__":
    main()
