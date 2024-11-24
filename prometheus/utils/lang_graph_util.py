from typing import Callable, Dict


def check_remaining_steps(
  state: Dict,
  router: Callable[..., str],
  min_remaining_steps: int,
  remaining_steps_key: str = "remaining_steps",
) -> str:
  original_route = router(state)
  if state[remaining_steps_key] > min_remaining_steps:
    return original_route
  else:
    return "low_remaining_steps"
