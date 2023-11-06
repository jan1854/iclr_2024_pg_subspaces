from pathlib import Path
from typing import Any, Dict, List, Optional

import stable_baselines3.common.logger

from pg_subspaces.metrics.tensorboard_logs import add_new_data_indicator


class SB3CustomLogger(stable_baselines3.common.logger.Logger):
    def __init__(
        self,
        folder: Optional[str],
        output_formats: List[stable_baselines3.common.logger.KVWriter],
    ):
        super().__init__(folder, output_formats)

    def dump(self, step: int = 0) -> None:
        if self.level == stable_baselines3.common.logger.DISABLED:
            return
        for _format in self.output_formats:
            if isinstance(_format, stable_baselines3.common.logger.KVWriter):
                _format.write(
                    self.name_to_value,
                    self.name_to_excluded,
                    step,
                )
        run_dir = Path(self.dir).parent
        add_new_data_indicator(run_dir)

        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    @staticmethod
    def _add_key_prefix(d: Dict[str, Any], prefix: str) -> Dict:
        return {f"{prefix}/{k}": v for k, v in d.items()}
