"""Run an observation-only RGB-D episode through the shared FLASH perception pipeline."""

from lerobot.policies.flash.observation_processing import RecordedMaskSource, extract, main

__all__ = ["RecordedMaskSource", "extract", "main"]

if __name__ == "__main__":
    main()
