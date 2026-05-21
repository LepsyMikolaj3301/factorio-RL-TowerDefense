-- Tower Defense scenario control.lua
-- Non-peaceful mode with enemy spawning support
util = require("util")

script.on_init(function()
    -- Enable enemies
    game.map_settings.enemy_expansion.enabled = false  -- We control spawning
    game.map_settings.enemy_evolution.enabled = true

    -- Track elapsed ticks
    storage.elapsed_ticks = 0
    storage.peaceful = false  -- Mark as non-peaceful so remove_enemies won't run

    -- Chart starting area
    local r = 200
    local force = game.forces.player
    local surface = game.surfaces[1]
    local origin = force.get_spawn_position(surface)
    force.chart(surface, {{origin.x - r, origin.y - r}, {origin.x + r, origin.y + r}})
end)

script.on_event(defines.events.on_tick, function(event)
    storage.elapsed_ticks = (storage.elapsed_ticks or 0) + 1
end)

script.on_event(defines.events.on_player_created, function(event)
    local player = game.get_player(event.player_index)
    if player and player.character then
        player.character.destructible = true
    end
end)
