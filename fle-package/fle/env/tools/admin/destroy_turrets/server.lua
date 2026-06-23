storage.actions.destroy_turrets = function(player_index, positions_str, radius)
    local player = storage.agent_characters[player_index] or game.get_player(player_index)
    if not player then
        rcon.print("0")
        return
    end
    local surface = player.surface
    radius = radius or 0.6

    -- positions_str comes as: {x=1.0,y=2.0}, {x=3.0,y=4.0}
    local positions = load("return {" .. positions_str .. "}")()

    local destroyed = 0
    for _, pos in ipairs(positions) do
        local turrets = surface.find_entities_filtered{
            -- gun-turret is of type "ammo-turret" in the Factorio API
            type = {"ammo-turret", "electric-turret", "fluid-turret"},
            area = {
                {pos.x - radius, pos.y - radius},
                {pos.x + radius, pos.y + radius},
            },
            force = "player",
        }
        for _, turret in ipairs(turrets) do
            if turret and turret.valid then
                turret.destroy()
                destroyed = destroyed + 1
            end
        end
    end

    rcon.print(serpent.line(destroyed))
end
