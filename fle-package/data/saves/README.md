# Tower-Defense save files

This folder holds the predefined Factorio `.zip` world(s) used by the tower-defense scenario.

The `.zip` saves are **gitignored** (`/data/saves/*.zip` in `.gitignore`) — they are large binaries and
must never be committed. Only this README is tracked, so the folder exists on a fresh clone.

## Default save

`fle cluster start -n 1` loads **`data/saves/tower_defense.zip`** by default (no `--save-path` needed).
Place your authored map here with exactly that name, or pass `--save-path <file>` to load a different
one, or `--generate` to build a fresh map from the scenario instead.

## Authoring a map

Build the world in the desktop Factorio client (version must match the server image,
**2.0.73**), starting **New Game → Scenarios → tower_defense** so the TD `control.lua` + charting are
baked in. Place a player-force **radar at (0,0)**, gun-turrets, and walls; place enemy-force
**spawners + biters** outside the wall ring; set evolution and chart the play area; then save and copy
the resulting `.zip` here as `tower_defense.zip`.

The agent character is created by the framework at **(0,10)** on connect — do not bake a player into the
save.
