from .slidevqa import SlideVQADataset
from .m3av import M3AVDataset
from .lpm import LPMDataset

DATASET_REGISTRY: dict[str, type] = {
    "slidevqa": SlideVQADataset,
    "m3av": M3AVDataset,
    "lpm": LPMDataset,
}
