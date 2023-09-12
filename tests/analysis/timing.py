from typing import Any, Callable
import timeit
import numpy as np


def bench_all(all_methods: list[tuple[Callable, str, Any]], repeats=1000):
    method1 = all_methods[0][0]
    for i, (method, name, reorder_output) in enumerate(all_methods):
        n = i + 1
        try:
            t = timeit.timeit(method, number=repeats)
            t = timeit.timeit(method, number=repeats)
        except Exception as e:
            print(e)
            continue
        extra = ""
        if i > 0:
            output = method()
            if reorder_output is not None:
                output = reorder_output(output)
            err = np.linalg.norm(method1() - output)
            extra = f" error={err:1.2g}"
        print(f"Method{n} {name}:\n time={t/repeats:5f}s" + extra)
