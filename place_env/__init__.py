from gym.envs.registration import register

register(
    id = "Place-v0",
    entry_point = "place_env:PlaceEnv",
)