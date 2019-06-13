"""
Values and Counter Utilities
"""
import math
from typing import Dict, List, Tuple, Callable, Optional, Union

__all__ = ['AnnealedValue', 'LambdaValue', 'MilestoneCounter',
           'Average', 'MovingAverage', 'WeightedAverage', 'SimpleAverage', 'HarmonicMean',
           'add_record', 'record_value', 'summarize_values']


class AnnealedValue:
    r"""
    Linearly interpolate values given keyframes. A keyframe is a tuple of `(iteration, value)`, and values for
    iterations in between keyframes are interpolated linearly.

    This class is typically used for annealed learning rates. It must be used in conjunction with
    :py:class:`nn.arguments.Arguments`.

    To construct an instance of this class, provide the keyframes as a list of `(iteration, value)` pairs. The begin
    and end keyframes are automatically inserted, namely:

    - `(0, init_val)` is inserted at the beginning.
    - `(+inf, final_val)` is inserted at the end.

    :param keyframes: List of `(iteration, value)` pairs.
    """

    def __init__(self, keyframes: List[Tuple[float, float]]):
        assert len(keyframes) > 0
        self.keyframes = list(sorted(keyframes))
        self.keyframes.insert(0, (0, self.keyframes[0][1]))
        self.keyframes.append((math.inf, self.keyframes[-1][1]))

        self.slopes: List[float] = []
        for i in range(len(self.keyframes) - 1):
            this_frame = self.keyframes[i]
            next_frame = self.keyframes[i + 1]
            if this_frame[1] == next_frame[1]:
                self.slopes.append(0)
            else:
                self.slopes.append(float(next_frame[1] - this_frame[1]) / (next_frame[0] - this_frame[0]))

        self.last_frame = 0  # cache last used frame, so calling the function monotonically would be faster

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This is not how you should use this class. "
                                  "Did you call `bind_to` on `Arguments`?")

    def value(self, iteration: int) -> float:
        r"""
        Return the value at a specific iteration.

        :param iteration: The iteration.
        """
        if self.keyframes[self.last_frame][0] > iteration:
            self.last_frame = 0
        while self.last_frame < len(self.keyframes):
            this_frame = self.keyframes[self.last_frame]
            next_frame = self.keyframes[self.last_frame + 1]
            if this_frame[0] <= iteration <= next_frame[0]:
                return this_frame[1] + self.slopes[self.last_frame] * (iteration - this_frame[0])
            self.last_frame += 1
        assert False


class LambdaValue:
    def __init__(self, func: Callable[[int], float]):
        self.func = func

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This is not how you should use this class. "
                                  "Did you call `bind_to` on `Arguments`?")

    def value(self, iteration: int) -> float:
        r"""
        Return the value at a specific iteration.

        :param iteration: The iteration.
        """
        return self.func(iteration)


class MilestoneCounter:
    r"""
    Equally distribute milestones according to progress. Keep track of current progress.
    """

    def __init__(self, total: int, *, scale: Optional[float] = None, milestones: Optional[int] = None):
        self.total = total
        if scale is not None:
            self.scale = float(scale) * self.total
        elif milestones is not None:
            self.scale = float(total) / milestones
        else:
            raise ValueError("'scale' and 'milestones' cannot both be None")
        self.progress_ = 0
        self.last_milestone = 0

    def progress(self, amount: int):
        self.progress_ += amount

    def milestone(self) -> int:
        r"""
        :return: How many milestones have passed since last query.
        """
        count = 0
        while self.last_milestone * self.scale < self.progress_:
            count += 1
            self.last_milestone += 1
        return count


class Average:
    def add(self, value: float):
        raise NotImplementedError

    def value(self) -> float:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class MovingAverage(Average):
    def __init__(self, length: int, plateau_threshold: int = 5):
        # Assume optimization has reached plateau if moving average does not decrease for 5 consecutive iterations
        self.length = length
        self.values: List[float] = []
        self.sum = 0.0
        self.previous_best = float('inf')
        self.plateau_iters = 0
        self.plateau_threshold = plateau_threshold

    def add(self, value: float):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.length:
            self.sum -= self.values.pop(0)

        val = self.value()
        if val < self.previous_best:
            self.previous_best = val
            self.plateau_iters = 0
        else:
            self.plateau_iters += 1

    def value(self) -> float:
        return float(self.sum) / min(len(self.values), self.length)

    def clear(self) -> None:
        self.values = []
        self.sum = 0
        self.previous_best = float('inf')
        self.plateau_iters = 0

    def decreasing(self) -> bool:
        return self.plateau_iters <= self.plateau_threshold

    def reset_stats(self) -> None:
        self.plateau_iters = 0


class HarmonicMean(Average):
    def __init__(self):
        self.values: List[float] = []

    def add(self, value: float):
        self.values.append(value)

    def value(self) -> float:
        if len(self.values) == 0 or any(val == 0 for val in self.values):
            return 0.0
        return 1.0 / (sum(1.0 / val for val in self.values) / len(self.values))

    def clear(self) -> None:
        self.values = []


class WeightedAverage(Average):
    def __init__(self):
        self.sum = 0.0
        self.count = 0.0

    def add(self, value: float, count: float = 1.0):
        self.sum += value * count
        self.count += count

    def value(self) -> float:
        return self.sum / self.count

    def clear(self) -> None:
        self.sum = 0.0
        self.count = 0.0


class SimpleAverage(Average):
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value: float):
        self.sum += value
        self.count += 1

    def value(self) -> float:
        return self.sum / self.count

    def clear(self) -> None:
        self.sum = 0.0
        self.count = 0


class _ValueRecords:
    value_dict: Dict[str, Average] = {}


def add_record(key: str, average_obj: Optional[Average] = None):
    if average_obj is None:
        average_obj = SimpleAverage()
    _ValueRecords.value_dict[key] = average_obj


def record_value(key: str, value: float, *args, **kwargs):
    _ValueRecords.value_dict[key].add(value, *args, **kwargs)  # type: ignore


def summarize_values(string: bool = True) -> Union[str, Dict[str, float]]:
    values = {}
    for key, records in _ValueRecords.value_dict.items():
        values[key] = records.value()
        records.clear()
    if string:
        return ', '.join([f'{k} = {v:f}' for k, v in values.items()])
    return values
