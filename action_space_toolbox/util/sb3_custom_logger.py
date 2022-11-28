from typing import Any, Dict, List, Optional

import stable_baselines3.common.logger


class SB3CustomLogger(stable_baselines3.common.logger.Logger):
    def __init__(
        self,
        folder: Optional[str],
        output_formats: List[stable_baselines3.common.logger.KVWriter],
        original_env_step_factor: int,
    ):
        super().__init__(folder, output_formats)
        self.original_env_step_factor = original_env_step_factor

    def dump(self, step: int = 0) -> None:
        if self.level == stable_baselines3.common.logger.DISABLED:
            return
        for _format in self.output_formats:
            if isinstance(_format, stable_baselines3.common.logger.KVWriter):
                _format.write(
                    self.name_to_value,
                    self.name_to_excluded,
                    step * self.original_env_step_factor,
                )
                _format.write(
                    self._add_key_prefix(self.name_to_value, "z_agent_timestep"),
                    self._add_key_prefix(self.name_to_excluded, "z_agent_timestep"),
                    step,
                )

        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    @staticmethod
    def _add_key_prefix(d: Dict[str, Any], prefix: str) -> Dict:
        return {f"{prefix}/{k}": v for k, v in d.items()}
