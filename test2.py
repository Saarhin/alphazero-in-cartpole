

import ray


@ray.remote
def my_task(x):
    y = x * x
    breakpoint()  # Add a breakpoint in the ray task
    return y


@ray.remote
def post_mortem(x):
    x += 1
    raise Exception("An exception is raised")
    return x