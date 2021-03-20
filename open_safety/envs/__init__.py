from gym.envs.registration import register

register(id="BalanceBotEnv-v0", \
        entry_point="open_safety.envs.balance_bot_env:BalanceBotEnv")
register(id="DuckBalanceBotEnv-v0", \
        entry_point="open_safety.envs.balance_bot_env:DuckBalanceBotEnv")
register(id="CubeBalanceBotEnv-v0", \
        entry_point="open_safety.envs.balance_bot_env:CubeBalanceBotEnv")
register(id="SphereBalanceBotEnv-v0", \
        entry_point="open_safety.envs.balance_bot_env:SphereBalanceBotEnv")

register(id="DuckMinitaurEnv-v0", \
        entry_point="open_safety.envs.minitaur_env:DuckMinitaurEnv")
register(id="CubeMinitaurEnv-v0", \
        entry_point="open_safety.envs.minitaur_env:CubeMinitaurEnv")
register(id="SphereMinitaurEnv-v0", \
        entry_point="open_safety.envs.minitaur_env:SphereMinitaurEnv")

register(id="DuckRacecarEnv-v0", \
        entry_point="open_safety.envs.racecar_env:DuckRacecarEnv")
register(id="CubeRacecarEnv-v0", \
        entry_point="open_safety.envs.racecar_env:CubeRacecarEnv")
register(id="SphereRacecarEnv-v0", \
        entry_point="open_safety.envs.racecar_env:SphereRacecarEnv")
