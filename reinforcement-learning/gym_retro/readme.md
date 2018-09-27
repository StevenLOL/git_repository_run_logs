

https://github.com/openai/retro



# install
pip3 install gym-retro
# test case A


```
What environments are there?

import retro
retro.data.list_games()
What initial states are there?

import retro
for game in retro.data.list_games():
    print(game, retro.data.list_states(game))
```
