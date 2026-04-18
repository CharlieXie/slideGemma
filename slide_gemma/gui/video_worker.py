"""QThread wrapper for the lecture-video analysis pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .i18n import normalize_ui_language, tr


class VideoAnalysisWorker(QThread):
    log_line = Signal(str)
    failed = Signal(str)
    completed = Signal(object)

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        server_url: str,
        language: str | None = "English",
        ui_language: str = "en",
    ):
        super().__init__()
        self._video_path = video_path
        self._output_dir = output_dir
        self._server_url = server_url
        self._language = language
        self._ui_language = normalize_ui_language(ui_language)
        self._running = True

    def run(self):
        try:
            result = self._run_pipeline()
            if self._running:
                self.completed.emit(result)
        except InterruptedError:
            self.log_line.emit(tr(self._ui_language, "video_analysis_stopped"))
        except Exception as exc:
            self.failed.emit(str(exc))

    def stop(self):
        self._running = False

    def _run_pipeline(self) -> dict:
        from ..models.loader import load_model, generate
        from ..media.frames import get_video_info, extract_frames
        from ..media.segments import adaptive_detect, time_based_segments
        from ..analysis.classifier import VideoType, classify_video_type
        from ..analysis.context import LectureContext
        from ..analysis.prompts import get_defaults_for_type
        from ..analysis.pipeline import analyze_segment, generate_summary
        from ..output.report import generate_report
        from ..output.video import compose_annotated_video
        from ..media.audio import extract_audio, transcribe, get_transcript_for_range

        ui = self._ui_language
        video_path = os.path.abspath(self._video_path)
        output_dir = os.path.abspath(self._output_dir)
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        def log(msg): self.log_line.emit(msg)
        def guard():
            if not self._running:
                raise InterruptedError(tr(ui, "video_analysis_stopped"))

        log(tr(ui, "reading_video_info"))
        vinfo = get_video_info(video_path)
        guard()

        log(tr(ui, "connecting_ai"))
        model, processor = load_model(server_url=self._server_url)
        guard()

        log(tr(ui, "extracting_frames", fps=1.0))
        frames, timestamps = extract_frames(video_path, fps=1.0)
        if not frames:
            raise RuntimeError(tr(ui, "no_frames_extracted"))
        guard()

        frame_paths = []
        for idx, frame in enumerate(frames):
            fp = os.path.join(frames_dir, f"frame_{idx:05d}.png")
            frame.save(fp)
            frame_paths.append(os.path.abspath(fp))
        guard()

        log(tr(ui, "recognizing_video_type"))
        video_type = classify_video_type(model, processor, frame_paths)
        log(tr(ui, "video_type", value=video_type.value))
        guard()

        defaults = get_defaults_for_type(video_type)
        log(tr(ui, "splitting_video_segments"))
        if video_type == VideoType.TEACHER_ONLY:
            segments = time_based_segments(frames, timestamps, interval=30.0)
        else:
            segments = adaptive_detect(
                frames, timestamps,
                initial_threshold=defaults["threshold"],
                min_duration=defaults["min_duration"],
                representative=defaults["representative"],
            )
        guard()

        for seg in segments:
            sp = os.path.join(frames_dir, f"slide_{seg.index:04d}.png")
            seg.representative_frame.save(sp)
            seg.frame_path = os.path.abspath(sp)

        analyses = []
        context = LectureContext(max_entries=10)
        for idx, seg in enumerate(segments):
            guard()
            log(tr(ui, "analyzing_segment", index=idx+1, total=len(segments), start=seg.start_time, end=seg.end_time))
            prev = segments[idx-1].frame_path if idx > 0 else None
            analysis = analyze_segment(
                model, processor, seg, context, video_type,
                total_segments=len(segments), prev_frame_path=prev,
                max_tokens=1024, language=self._language,
            )
            analyses.append(analysis)
            context.add(seg.index, LectureContext.extract_summary(analysis))

        guard()
        log(tr(ui, "generating_summary"))
        summary = generate_summary(model, processor, analyses, max_tokens=1024)

        log(tr(ui, "writing_report"))
        report_path = generate_report(
            video_path, vinfo, video_type, segments, analyses, summary, output_dir,
        )
        guard()

        log(tr(ui, "composing_video"))
        output_video = os.path.join(output_dir, "annotated_video.mp4")
        compose_annotated_video(video_path, segments, analyses, output_video)

        return {
            "video_path": video_path, "output_dir": output_dir,
            "report_path": report_path, "output_video": output_video,
            "segments": len(segments), "video_type": video_type.value,
            "title": Path(video_path).name,
        }
