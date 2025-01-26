from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Iterable, Any
import json
import time
import gc
import warnings
import numpy as np


@dataclass
class BenchmarkItem:
    """Benchmark measurements for one operation.

    This class records the measurements of execution times for different problem sizes.
    """

    name: str
    sizes: List[int]
    times: List[float]

    def tojson(self):
        return {
            "name": self.name,
            "sizes": self.sizes,
            "times": self.times,
        }

    @classmethod
    def fromjson(cls, data):
        return BenchmarkItem(
            name=data["name"], sizes=data["sizes"], times=data["times"]
        )

    @classmethod
    def timeit(cls, function: Callable, number: int, withgc: bool = True):
        """Execute `function` a `number` of times and return time taken in seconds."""
        if withgc:
            gc.collect()
            t = time.perf_counter()
            for _ in range(number):
                function()
            t = time.perf_counter() - t
        else:
            gcold = gc.isenabled()
            gc.disable()
            try:
                t = time.perf_counter()
                for _ in range(number):
                    function()
                t = time.perf_counter() - t
            finally:
                if gcold:
                    gc.enable()
        return t

    @classmethod
    def autorange(cls, function: Callable, limit: float = 0.2):
        number: int = 1
        time_taken: float = 0.0
        for _ in range(10):
            time_taken = cls.timeit(function, number)
            # print(f"time_taken = {time_taken} vs limit = {limit}")
            if time_taken >= limit:
                break
            number = max(round(1.5 * limit / time_taken * number), 1)
            # print(f"trying again with number = {number}")
        return time_taken / number

    @staticmethod
    def run(
        name: str,
        function: Callable,
        setup: Optional[Callable[[int], tuple]] = None,
        sizes: Optional[List[int]] = None,
        limit: float = 0.2,
    ):
        if sizes == None:
            sizes = [2**i for i in range(1, 7)]
        times = []
        for s in sizes:
            args: Iterable[Any]
            if setup is None:
                args = []
            else:
                args = setup(s)
            timing = BenchmarkItem.autorange(lambda: function(*args), limit)
            times.append(timing)
            print(f"Executing item {name} at size {s} took {timing:5g} seconds")
        return BenchmarkItem(name=name, sizes=sizes, times=times)


@dataclass
class BenchmarkGroup:
    name: str
    items: List[BenchmarkItem]

    @staticmethod
    def run(name: str, items: List[Tuple[str, Callable, Callable]]) -> "BenchmarkGroup":
        print("-" * 50)
        print(f"Executing group {name}")
        return BenchmarkGroup(
            name=name, items=[BenchmarkItem.run(*item) for item in items]
        )

    def maybe_find_item(
        self, name: str, error: bool = False
    ) -> Optional[BenchmarkItem]:
        for it in self.items:
            if it.name == name:
                return it
        return None

    def find_item(self, name: str) -> BenchmarkItem:
        item = self.maybe_find_item(name)
        if item is None:
            raise Exception(f"Item {name} missing from group {self.name}")
        return item

    def tojson(self):
        return {
            "name": self.name,
            "items": [item.tojson() for item in self.items],
        }

    @classmethod
    def fromjson(cls, data):
        name = data["name"]
        items = [BenchmarkItem.fromjson(item) for item in data["items"]]
        return BenchmarkGroup(name=name, items=items)


@dataclass
class BenchmarkSet:
    name: str
    groups: List[BenchmarkGroup]
    environment: str

    def write(self, filename: Optional[str] = None):
        if not filename:
            filename = self.name + ".json"
        with open(filename, "w") as f:
            json.dump(self.tojson(), f)

    def tojson(self):
        return {
            "name": self.name,
            "environment": self.environment,
            "groups": [item.tojson() for item in self.groups],
        }

    def maybe_find_group(self, name: str) -> Optional[BenchmarkGroup]:
        for g in self.groups:
            if g.name == name:
                return g
        return None

    def find_group(self, name: str) -> BenchmarkGroup:
        group = self.maybe_find_group(name)
        if group is None:
            raise Exception(f"Group {name} missing from benchmark {self.name}")
        return group

    @classmethod
    def fromjson(cls, data) -> "BenchmarkSet":
        return BenchmarkSet(
            name=data["name"],
            environment=data["environment"],
            groups=[BenchmarkGroup.fromjson(item) for item in data["groups"]],
        )

    @staticmethod
    def fromjson_file(filename: str) -> "BenchmarkSet":
        with open(filename, "r") as f:
            return BenchmarkSet.fromjson(json.load(f))

    @staticmethod
    def find_all_pairs(benchmarks: List["BenchmarkSet"]) -> List[Tuple[str, str]]:
        output = set()
        for b in benchmarks:
            for g in b.groups:
                for i in g.items:
                    output.add((g.name, i.name))
        return sorted(list(output), key=lambda p: "\b".join(p))


@dataclass
class BenchmarkItemAggregate:
    columns: List[str]
    sizes: List[int]
    times: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def __init__(self, benchmarks: List[BenchmarkSet], group_name: str, item_name: str):
        if not benchmarks:
            return
        items = []
        valid = []
        for set in benchmarks:
            item = None
            group = set.maybe_find_group(group_name)
            if group is not None:
                item = group.maybe_find_item(item_name)
            if item is not None:
                items.append(item)
                valid.append(set)
            else:
                warnings.warn(
                    f"Benchmark set {set.name} lacks group {group_name} or item {item_name}"
                )
        self.columns = [b.name for b in valid]
        self.sizes = items[0].sizes
        for n, set in enumerate(valid):
            if not np.all(items[n].sizes == self.sizes):
                raise Exception(
                    f"Benchmark set {set.name} has differring sizes for group {group_name} and item {item_name}"
                )
        self.times = np.array([i.times for i in items])
        if group_name == "RTensor" and item_name == "plus":
            for set in benchmarks:
                print(set.name)
                group = set.find_group(group_name)
                item = group.find_item(item_name)
                print(item.times)
            print(self.columns)
            print(self.times)
