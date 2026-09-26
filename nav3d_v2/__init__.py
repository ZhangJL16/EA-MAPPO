"""Independent path-tracking variant of the frozen v1 analytic navigator."""

from .controller import SegmentTrackingController, fly_segment_route

__all__ = ["SegmentTrackingController", "fly_segment_route"]
