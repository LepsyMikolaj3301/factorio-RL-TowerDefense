-- Tower Defense scenario control.lua
-- Non-peaceful mode with enemy spawning support
--
-- Map authoring notes:
--   * The env READS the existing radar at (0,0) — it never places one.
--   * Enemy spawners (unit-spawner, enemy force) should be baked into the save
--     outside the wall ring; they persist across all resets for free.
--   * on_init runs only when a brand-new map is created, NOT when loading a save.
--     The RL env always loads a prebuilt save, so all persistent state below is
--     lazily initialized in td_init() (guarded) rather than relying on on_init.
util = require("util")

-- ---------------------------------------------------------------------------
-- Lazy persistent-state init. Safe to call every tick; only fills nil fields.
-- ---------------------------------------------------------------------------
local function td_init()
    if storage.elapsed_ticks == nil then storage.elapsed_ticks = 0 end
    -- Reward event counters (read-and-cleared by the admin/read_td_events tool).
    if storage.td_events == nil then
        storage.td_events = {
            kills = 0,
            turrets_lost = 0,
            walls_lost = 0,
            buildings_lost = 0,
            boilers_lost = 0,
            radar_lost = 0,
            char_died = 0,
        }
    end
    -- Async A* walk controller (driven from Python via agent/walk_to).
    if storage.td_walk == nil then
        storage.td_walk = {active = false, path = {}, idx = 1, target_anchor = -1, speed = 0.2}
    end
    -- Previous-frame group centroids for cheap velocity estimation in threat_view.
    if storage.td_threat_prev == nil then
        storage.td_threat_prev = {tick = 0, groups = {}}
    end
end

script.on_init(function()
    -- Enable enemies
    game.map_settings.enemy_expansion.enabled = false  -- We control spawning
    game.map_settings.enemy_evolution.enabled = true
    storage.peaceful = false  -- Mark as non-peaceful so remove_enemies won't run
    td_init()

    -- Chart starting area
    local r = 200
    local force = game.forces.player
    local surface = game.surfaces[1]
    local origin = force.get_spawn_position(surface)
    force.chart(surface, {{origin.x - r, origin.y - r}, {origin.x + r, origin.y + r}})
end)

-- ---------------------------------------------------------------------------
-- Single on_tick handler: tick counter + async A* walker.
-- (Registering on_tick twice would REPLACE the first handler, so both pieces
--  must live here together.)
-- ---------------------------------------------------------------------------
script.on_event(defines.events.on_tick, function(event)
    td_init()
    storage.elapsed_ticks = (storage.elapsed_ticks or 0) + 1

    local walk = storage.td_walk
    if not walk.active then return end

    local char = storage.agent_characters and storage.agent_characters[1]
    if not (char and char.valid) then
        walk.active = false
        return
    end

    local path = walk.path
    if not path or #path == 0 or walk.idx > #path then
        walk.active = false
        return
    end

    local speed = walk.speed or 0.2
    local remaining = speed
    -- Advance along the polyline up to `speed` tiles this tick.
    while remaining > 0 and walk.idx <= #path do
        local wp = path[walk.idx]
        local px, py = char.position.x, char.position.y
        local dx, dy = wp.x - px, wp.y - py
        local dist = math.sqrt(dx * dx + dy * dy)
        if dist <= remaining then
            char.teleport({wp.x, wp.y})
            walk.idx = walk.idx + 1
            remaining = remaining - dist
        else
            local t = remaining / dist
            char.teleport({px + dx * t, py + dy * t})
            remaining = 0
        end
    end

    if walk.idx > #path then
        walk.active = false
    end
end)

-- ---------------------------------------------------------------------------
-- Reward event counters. Classify every relevant death; Python reads & clears
-- the counters each step via admin/read_td_events. This accurately attributes
-- enemy kills (all biter/spitter types — fixes the old small-biter-only bug)
-- and structure losses (turrets / walls / buildings / radar / character).
-- ---------------------------------------------------------------------------
script.on_event(defines.events.on_entity_died, function(event)
    td_init()
    local ent = event.entity
    if not ent then return end
    local ev = storage.td_events
    local fname = ent.force and ent.force.name or ""
    local etype = ent.type
    local ename = ent.name

    if fname == "enemy" then
        if etype == "unit" then
            ev.kills = ev.kills + 1
        end
        return
    end

    if fname == "player" then
        if etype == "ammo-turret" or etype == "electric-turret"
            or etype == "fluid-turret" or ename == "gun-turret" then
            ev.turrets_lost = ev.turrets_lost + 1
        elseif etype == "wall" or etype == "gate" then
            ev.walls_lost = ev.walls_lost + 1
        elseif etype == "radar" then
            ev.radar_lost = ev.radar_lost + 1
        elseif ename == "boiler" then
            ev.boilers_lost = (ev.boilers_lost or 0) + 1
            ev.buildings_lost = ev.buildings_lost + 1
        elseif etype == "character" then
            ev.char_died = 1
        else
            -- Any other player-force structure inside the perimeter.
            ev.buildings_lost = ev.buildings_lost + 1
        end
    end
end)

script.on_event(defines.events.on_player_created, function(event)
    local player = game.get_player(event.player_index)
    if player and player.character then
        player.character.destructible = true
    end
end)
