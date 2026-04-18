#!/usr/bin/env python3
"""Lecture video analysis CLI -- powered by Gemma 4.

Usage::

    python tools/analyze.py VIDEO_PATH [options]

Run ``python tools/analyze.py -h`` for the full option list.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger("slide_gemma.analyze")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse a lecture video with Gemma 4 and generate a report.",
    )
    p.add_argument("video", help="Path to the lecture video file")

    g_model = p.add_argument_group("model")
    g_model.add_argument("--model", choices=["e2b", "e4b"], default="e2b",
                         help="Gemma 4 model variant (default: e2b)")
    g_model.add_argument("--local-model-dir", default=None,
                         help="Path to a pre-downloaded model directory")
    g_model.add_argument("--gpu", type=int, default=0, help="CUDA device id (default: 0)")
    g_model.add_argument("--max-tokens", type=int, default=1024,
                         help="Max generation tokens per segment (default: 1024)")

    g_video = p.add_argument_group("video analysis")
    g_video.add_argument("--mode", default="auto",
                         choices=["auto", "slides", "teacher_slides", "whiteboard",
                                  "teacher_only", "screen_recording"],
                         help="Video type (default: auto-detect)")
    g_video.add_argument("--fps", type=float, default=1.0,
                         help="Frame extraction rate in fps (default: 1.0)")
    g_video.add_argument("--threshold", type=float, default=None,
                         help="Scene-change threshold 0-1 (default: auto per mode)")
    g_video.add_argument("--min-duration", type=float, default=None,
                         help="Min seconds between segments (default: auto per mode)")

    g_audio = p.add_argument_group("audio (optional)")
    g_audio.add_argument("--audio", action="store_true",
                         help="Enable audio transcription via Whisper")
    g_audio.add_argument("--whisper-model", default="base",
                         help="Whisper model size (default: base)")

    g_out = p.add_argument_group("output")
    g_out.add_argument("--output-dir", default="./output",
                       help="Output directory (default: ./output)")
    g_out.add_argument("--language", default=None,
                       help="Force output language (default: auto-detect from content)")
    g_out.add_argument("--realtime", action="store_true",
                       help="Print each analysis to stdout as it is generated")
    g_out.add_argument("-v", "--verbose", action="store_true",
                       help="Enable debug logging")

    return p.parse_args(argv)


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        sys.exit(f"Video not found: {video_path}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from slide_gemma.models import load_model
    from slide_gemma.media import (
        get_video_info, extract_frames,
        detect_segments, time_based_segments, adaptive_detect,
        extract_audio, transcribe, get_transcript_for_range,
    )
    from slide_gemma.analysis import (
        VideoType, LectureContext, classify_video_type,
        analyze_segment, generate_summary, get_defaults_for_type,
    )
    from slide_gemma.output import generate_report, format_timestamp, compose_annotated_video

    output_dir = os.path.abspath(args.output_dir)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    banner = "=" * 60
    print(f"\n{banner}")
    print("  Lecture-Lens  (Gemma 4)")
    print(f"{banner}\n")

    # 1. Video metadata
    vinfo = get_video_info(video_path)
    print(f"Video   : {video_path}")
    print(f"Duration: {format_timestamp(vinfo['duration'])}  |  "
          f"FPS: {vinfo['fps']:.1f}  |  "
          f"Size: {vinfo['width']}x{vinfo['height']}  |  "
          f"Audio: {'yes' if vinfo['has_audio'] else 'no'}")

    # 2. Load model
    print(f"\n[1/6] Loading Gemma 4 ({args.model.upper()}) ...")
    model, processor = load_model(args.model, args.local_model_dir)

    # 3. Extract frames
    print(f"\n[2/6] Extracting frames at {args.fps} fps ...")
    frames, timestamps = extract_frames(video_path, fps=args.fps)
    print(f"       {len(frames)} frames extracted")
    if not frames:
        sys.exit("No frames could be extracted from the video.")

    frame_paths: list[str] = []
    for i, frame in enumerate(frames):
        p = os.path.join(frames_dir, f"frame_{i:05d}.png")
        frame.save(p)
        frame_paths.append(os.path.abspath(p))

    # 4. Classify video type
    if args.mode == "auto":
        print("\n[3/6] Classifying video type ...")
        video_type = classify_video_type(model, processor, frame_paths)
    else:
        video_type = VideoType(args.mode.upper())
    print(f"       Type: {video_type.value}")

    # 5. Detect segments
    defaults = get_defaults_for_type(video_type)
    threshold = args.threshold if args.threshold is not None else defaults["threshold"]
    min_dur = args.min_duration if args.min_duration is not None else defaults["min_duration"]
    rep = defaults["representative"]

    print(f"\n[4/6] Detecting segments (threshold={threshold:.3f}, "
          f"min_duration={min_dur:.1f}s) ...")

    if video_type == VideoType.TEACHER_ONLY:
        segments = time_based_segments(frames, timestamps, interval=30.0)
    else:
        segments = adaptive_detect(
            frames, timestamps,
            initial_threshold=threshold,
            min_duration=min_dur,
            representative=rep,
        )

    for seg in segments:
        seg_path = os.path.join(frames_dir, f"slide_{seg.index:04d}.png")
        seg.representative_frame.save(seg_path)
        seg.frame_path = os.path.abspath(seg_path)

    print(f"       {len(segments)} segments detected")

    # 6. Audio transcription (optional)
    transcription: list[dict] | None = None
    full_transcript: str | None = None

    if args.audio:
        print("\n[5/6] Transcribing audio ...")
        if not vinfo["has_audio"]:
            print("       WARNING: video has no audio track -- skipping")
        else:
            audio_path = os.path.join(output_dir, "audio.wav")
            try:
                extract_audio(video_path, audio_path)
                transcription = transcribe(audio_path, model_size=args.whisper_model)
                full_transcript = " ".join(s["text"] for s in transcription)
                print(f"       {len(transcription)} transcript segments")
            except ImportError as exc:
                print(f"       WARNING: {exc}")
            except Exception as exc:
                logger.warning("Audio transcription failed: %s", exc)
                print(f"       WARNING: transcription failed -- {exc}")
    else:
        print("\n[5/6] Audio transcription skipped (use --audio to enable)")

    # 7. Analyse each segment
    print(f"\n[6/6] Analysing {len(segments)} segments ...")
    context = LectureContext(max_entries=10)
    analyses: list[str] = []

    for i, seg in enumerate(segments):
        tag = f"[{i + 1}/{len(segments)}]"
        print(f"\n{tag} Segment {seg.index + 1}  "
              f"({_ts(seg.start_time)} -- {_ts(seg.end_time)}) ...", flush=True)

        audio_text = None
        if transcription:
            audio_text = get_transcript_for_range(
                transcription, seg.start_time, seg.end_time
            ) or None

        prev_frame = segments[i - 1].frame_path if i > 0 else None

        analysis = analyze_segment(
            model, processor, seg, context, video_type,
            total_segments=len(segments),
            audio_text=audio_text,
            prev_frame_path=prev_frame,
            max_tokens=args.max_tokens,
            language=args.language,
        )
        analyses.append(analysis)
        context.add(seg.index, LectureContext.extract_summary(analysis))

        if args.realtime:
            print(f"\n{'_' * 50}")
            print(analysis)
            print(f"{'_' * 50}")

    # 8. Overall summary
    print("\nGenerating overall summary ...")
    summary = generate_summary(model, processor, analyses, max_tokens=args.max_tokens)

    # 9. Report
    report_path = generate_report(
        video_path, vinfo, video_type, segments, analyses, summary,
        output_dir, transcript_text=full_transcript,
    )

    # 10. Annotated video
    output_video = os.path.join(output_dir, "annotated_video.mp4")
    print("\nComposing annotated video ...")
    compose_annotated_video(video_path, segments, analyses, output_video)

    print(f"\n{banner}")
    print(f"  Done!")
    print(f"  Report : {report_path}")
    print(f"  Video  : {output_video}")
    print(f"  Frames : {frames_dir}/")
    print(f"{banner}\n")


if __name__ == "__main__":
    main()
