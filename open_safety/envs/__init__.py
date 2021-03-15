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
        entry_point="open_safety.envs.minitaur_env:MinitaurDuckBulletEnv")
register(id="CubeMinitaurEnv-v0", \
        entry_point="open_safety.envs.minitaur_env:MinitaurCubeBulletEnv")
register(id="SphereMinitaurEnv-v0", \
        entry_point="open_safety.envs.minitaur_env:MinitaurSphereBulletEnv")
