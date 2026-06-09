storage.actions.grant_items = function(player_index, items_str)
    local player = game.get_player(player_index)
    if not player or not player.character then
        rcon.print("No character for player " .. tostring(player_index))
        return
    end

    local inventory = player.get_main_inventory()
    if not inventory then
        rcon.print("No inventory")
        return
    end

    -- Parse items string into table
    -- items_str comes as: ["firearm-magazine"] = 20, ["stone-wall"] = 10
    local items = load("return {" .. items_str .. "}")()

    local granted = {}
    local max_per_item = 200  -- Cap to prevent infinite accumulation

    for item_name, count in pairs(items) do
        local current = inventory.get_item_count(item_name)
        local to_grant = math.min(count, math.max(0, max_per_item - current))
        if to_grant > 0 then
            local inserted = inventory.insert{name = item_name, count = to_grant}
            granted[item_name] = inserted
        end
    end

    rcon.print(serpent.line(granted))
end
