import ray
import sys
from test2 import my_task, post_mortem
# Add RAY_DEBUG environment variable to enable Ray Debugger
ray.init(
    runtime_env={
        "env_vars": {"RAY_DEBUG": "1"},
    }
)

print(len(sys.argv))
if len(sys.argv) == 1:
    ray.get(my_task.remote(10))



ray.get(post_mortem.remote(10))
