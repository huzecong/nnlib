from typing import Any, Callable, Dict, Generic, Iterable, List, NamedTuple, Optional, TypeVar, Union, no_type_check

from nnlib import utils
from nnlib.utils.logging import Logging

__all__ = ['Trainer']


class RecordList:
    def __init__(self):
        self.list = []

    def add(self, obj):
        self.list.append(obj)

    def value(self) -> List:
        return self.list

    def clear(self):
        self.list.clear()


PostComputeFunction = Union[Callable[[float], float], Callable[[float], str], Callable[[List], List]]


class TrainerRecord(NamedTuple):
    record: Union[RecordList, utils.Average]
    period: str
    precision: int
    display: str
    post_compute: Optional[PostComputeFunction]


T = TypeVar('T')


class Trainer(Generic[T]):
    r"""
    Generalized training procedure.
    """

    LOG_MESSAGE = "Epoch {epoch}, Iter {iter}: {records}"

    def __init__(self, *, valid_iters: int = 5000, log_iters: int = 10, max_epochs: int = -1, max_iters: int = -1,
                 decay_threshold: int = 3, patience: int = 5, metric_higher_better: bool = True,
                 timestamp: bool = True) -> None:
        self.epoch = 0
        self.iterations = 0
        self.valid_iters = valid_iters
        self.log_iters = log_iters
        self.max_epochs = max_epochs
        self.max_iters = max_iters
        self.timestamp = timestamp

        self.records: Dict[str, TrainerRecord] = {}
        self.before_iteration_hooks: List[Callable[[Trainer], None]] = []
        self.after_iteration_hooks: List[Callable[[Trainer], None]] = []

        self.validation_history: List[float] = []
        self.bad_counter = 0  # number of consecutive validations that did not improve
        self.decay_times = 0  # number of times that learning rate has decayed
        self.decay_threshold = decay_threshold  # decay learning rate when `bad_counter` reaches `decay_threshold`
        self.patience = patience  # early stop when `decay_times` reaches `patience`
        self.metric_higher_better = metric_higher_better  # whether higher evaluation metric is better

    def register_record(self, name: str, record_type: str = 'weighted', period: str = 'validate',
                        precision: int = 6, display: Optional[str] = None,
                        post_compute: Optional[PostComputeFunction] = None):
        r"""
        :param name: Name of the record.
        :param record_type: Choices: 'weighted', 'average', 'list'.
        :param period: When to clear recorded values. Choices: 'log', 'validate', 'epoch'.
        :param precision: Floating point precision when printing.
        :param display: The name to display when summarizing values.
        :param post_compute: Post computation to apply before displaying results (e.g. formatting, calculations).
        """
        if name in self.records:
            raise ValueError(f"Record with name '{name}' already exists")
        if period not in ['log', 'validate', 'epoch']:
            raise ValueError(f"Invalid `record_period`: '{period}'")

        # noinspection PyUnusedLocal
        value: Union[RecordList, utils.Average]
        if record_type == 'weighted':
            value = utils.WeightedAverage()
        elif record_type == 'average':
            value = utils.SimpleAverage()
        elif record_type == 'list':
            value = RecordList()
        else:
            raise ValueError(f"Invalid `record_type`: '{record_type}'")

        self.records[name] = TrainerRecord(record=value, period=period, precision=precision,
                                           display=display or name, post_compute=post_compute)

    def create_data_iterator(self) -> Iterable[T]:
        raise NotImplementedError

    def train_step(self, batch: T) -> Dict[str, Any]:
        raise NotImplementedError

    def validate(self) -> float:
        raise NotImplementedError

    def save_model(self) -> None:
        raise NotImplementedError

    def decay(self) -> None:
        raise NotImplementedError

    @no_type_check
    def _print_summary(self, period: str = 'log') -> None:
        summary = []
        for record in self.records.values():
            if record.period == period:
                value = record.record.value()
                if record.post_compute is not None:
                    value = record.post_compute(value)
                if isinstance(value, float):
                    value = f'{value:.{record.precision}f}'
                summary.append((record.display, value))
                record.record.clear()
        if len(summary) == 0:
            return
        records = ', '.join(f'{name}={value}' for name, value in summary)
        log_message = self.LOG_MESSAGE.format(epoch=self.epoch, iter=self.iterations, records=records)
        Logging(1).log(log_message, timestamp=self.timestamp)

    def train(self) -> None:
        # TODO: Incorporate `torchutils.prevent_oom`
        Logging(1).log("Training start.", timestamp=self.timestamp)
        while self.max_epochs == -1 or self.epoch < self.max_epochs:
            iterator = self.create_data_iterator()

            for batch in iterator:
                self.iterations += 1

                for hook in self.before_iteration_hooks:
                    hook(self)

                record_values = self.train_step(batch)

                for name, value in record_values.items():
                    if isinstance(value, tuple):
                        self.records[name].record.add(*value)
                    else:
                        self.records[name].record.add(value)

                for hook in self.after_iteration_hooks:
                    hook(self)

                if self.iterations % self.log_iters == 0:
                    self._print_summary(period='log')

                if self.iterations % self.valid_iters == 0:
                    self._print_summary(period='validate')

                    metric = self.validate()
                    if len(self.validation_history) == 0 or \
                            (metric > max(self.validation_history) and self.metric_higher_better) or \
                            (metric < min(self.validation_history) and not self.metric_higher_better):
                        self.bad_counter = 0
                        self.save_model()
                    else:
                        self.bad_counter += 1
                        Logging(1).log(f"{utils.ordinal(self.bad_counter)} time degradation "
                                       f"(threshold={self.decay_threshold}).")
                        if self.bad_counter >= self.decay_threshold:
                            self.decay_times += 1
                            if self.decay_times > self.patience:
                                Logging(1).log("Early stop!", color='red')
                                return
                            self.bad_counter = 0
                            self.decay()

                    self.validation_history.append(metric)

            self.epoch += 1
            self._print_summary(period='epoch')
