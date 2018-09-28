

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
# Install game Roms
## Donwload Roms
http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html
and extract All
## Import Roms
```
python -m retro.import Your_Rom_Path
```

REF: https://github.com/openai/retro/issues/60
