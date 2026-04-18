"""Media processing: frames, segments, and audio.

Submodules are imported lazily to avoid pulling in heavy dependencies
(av, moviepy) when only a subset of functionality is needed.
"""


def __getattr__(name):
    if name == "Segment":
        from .segments import Segment
        return Segment
    if name in ("detect_segments", "time_based_segments", "adaptive_detect"):
        from . import segments
        return getattr(segments, name)
    if name in ("get_video_info", "extract_frames"):
        from . import frames
        return getattr(frames, name)
    if name in ("extract_audio", "transcribe", "get_transcript_for_range"):
        from . import audio
        return getattr(audio, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
