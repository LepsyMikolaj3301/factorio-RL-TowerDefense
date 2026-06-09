storage.actions.shoot = function(player_index, target_x, target_y, duration_ticks)
    local player = game.get_player(player_index)
    if not player or not player.character then
        rcon.print("No character found for player " .. tostring(player_index))
        return
    end

    local character = player.character
    duration_ticks = duration_ticks or 30

    -- Set shooting state towards target
    character.shooting_state = {
        state = defines.shooting.shooting_enemies,
        position = {x = target_x, y = target_y}
    }

    -- Schedule stop after duration_ticks using on_nth_tick if available,
    -- otherwise just set the state (the env will manage tick stepping)
    rcon.print("shooting")
end
