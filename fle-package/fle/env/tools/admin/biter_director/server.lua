storage.actions.biter_director = function(player_index, wave_number, center_x, center_y, spawn_radius, base_count, escalation_factor, use_spawners, target_x, target_y)
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

    -- Default attack destination to the defended center if not explicitly given
    local dest_x = target_x or center_x
    local dest_y = target_y or center_y

    -- Collect spawn origins: baked-in spawners when use_spawners=true, else
    -- fall back to a geometric ring around (center_x, center_y).
    local origins = {}
    if use_spawners then
        local spawners = surface.find_entities_filtered{type = "unit-spawner", force = "enemy"}
        for _, sp in ipairs(spawners) do
            table.insert(origins, {x = sp.position.x, y = sp.position.y})
        end
    end
    -- Fall back to geometric ring when no spawners were found
    if #origins == 0 then
        for i = 1, math.max(count, 1) do
            local angle = (i / math.max(count, 1)) * 2 * math.pi + math.random() * 0.3
            table.insert(origins, {
                x = center_x + math.cos(angle) * spawn_radius,
                y = center_y + math.sin(angle) * spawn_radius,
            })
        end
    end

    for i = 1, count do
        -- Round-robin across origins
        local origin = origins[((i - 1) % #origins) + 1]
        local pos = surface.find_non_colliding_position(enemy_type, origin, 5, 0.5)
        if not pos then
            pos = origin
        end

        local etype = enemy_type
        if include_spitters and i % 3 == 0 then
            etype = spitter_type
        end

        local entity = surface.create_entity{
            name = etype,
            position = pos,
            force = "enemy",
        }
        if entity then
            entity.set_command{
                type = defines.command.attack_area,
                destination = {x = dest_x, y = dest_y},
                radius = spawn_radius * 0.5,
            }
            spawned = spawned + 1
        end
    end
    rcon.print(spawned)
end
