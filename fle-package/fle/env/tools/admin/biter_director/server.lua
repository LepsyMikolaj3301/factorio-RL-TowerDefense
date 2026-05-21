storage.actions.biter_director = function(player_index, wave_number, center_x, center_y, spawn_radius, base_count, escalation_factor)
    local surface = game.surfaces[1]
    local count = math.floor(base_count * math.pow(escalation_factor, wave_number - 1))
    local spawned = 0

    -- Choose enemy type based on wave
    local enemy_type = "small-biter"
    if wave_number >= 10 then
        enemy_type = "big-biter"
    elseif wave_number >= 5 then
        enemy_type = "medium-biter"
    end

    -- Mix in spitters after wave 3
    local include_spitters = wave_number >= 3
    local spitter_type = "small-spitter"
    if wave_number >= 10 then
        spitter_type = "big-spitter"
    elseif wave_number >= 5 then
        spitter_type = "medium-spitter"
    end

    for i = 1, count do
        -- Distribute around a ring
        local angle = (i / count) * 2 * math.pi + math.random() * 0.3
        local x = center_x + math.cos(angle) * spawn_radius
        local y = center_y + math.sin(angle) * spawn_radius

        local etype = enemy_type
        if include_spitters and i % 3 == 0 then
            etype = spitter_type
        end

        local entity = surface.create_entity{
            name = etype,
            position = {x, y},
            force = "enemy",
        }
        if entity then
            -- Command them to attack the center
            entity.set_command{
                type = defines.command.attack_area,
                destination = {x = center_x, y = center_y},
                radius = spawn_radius * 0.5,
            }
            spawned = spawned + 1
        end
    end
    rcon.print(spawned)
end
