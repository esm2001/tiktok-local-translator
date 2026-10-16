"""Browser-independent contract; providers discover candidates, never select one."""

from abc import ABC, abstractmethod
from pathlib import Path


class ReverseSearchProvider(ABC):
    @abstractmethod
    def search(self, clean_path: Path, force: bool = False) -> dict:
        """Return URLs for raw downloads and backend visual ranking."""

    def close(self) -> None:
        """Release optional browser resources."""

    def set_deadline(self, deadline: float) -> None:
        """Optional monotonic job deadline; implementations cap external waits."""

    def search_text(self, text: str, force: bool = False) -> dict:
        """Optional final discovery step; candidates must decode as real images."""
        return {'provider': type(self).__name__, 'status': 'UNAVAILABLE', 'candidates': [],
                'error': 'This provider does not support text-assisted discovery.'}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
