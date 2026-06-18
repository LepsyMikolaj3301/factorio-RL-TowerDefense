storage.actions.spawn_enemy = function(player_index, entity_type, x, y, count)
    local surface = game.surfaces[1]
    local spawned = 0
    local valid_types = {
        ["small-biter"] = true, ["medium-biter"] = true,
        ["big-biter"] = true, ["behemoth-biter"] = true,
        ["small-spitter"] = true, ["medium-spitter"] = true,
        ["big-spitter"] = true, ["behemoth-spitter"] = true,
        ["biter-spawner"] = true, ["spitter-spawner"] = true,
    }

    if not valid_types[entity_type] then
        return "Invalid enemy type: " .. tostring(entity_type)
    end

    count = count or 1
    for i = 1, count do
        -- Offset slightly so they don't stack
        local offset_x = x + (i - 1) * 0.5
        local offset_y = y
        local entity = surface.create_entity{
            name = entity_type,
            position = {offset_x, offset_y},
            force = "enemy",
        }
        if entity then
            spawned = spawned + 1
        end
    end
    return spawned
end
