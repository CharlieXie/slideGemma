"""Analysis pipeline: classification, prompts, context, and segment analysis.

Submodules are imported lazily to avoid pulling in the full media stack
when only analysis utilities are needed.
"""


def __getattr__(name):
    if name in ("VideoType", "classify_video_type"):
        from . import classifier
        return getattr(classifier, name)
    if name == "LectureContext":
        from .context import LectureContext
        return LectureContext
    if name == "get_defaults_for_type":
        from .prompts import get_defaults_for_type
        return get_defaults_for_type
    if name in ("analyze_segment", "generate_summary"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
