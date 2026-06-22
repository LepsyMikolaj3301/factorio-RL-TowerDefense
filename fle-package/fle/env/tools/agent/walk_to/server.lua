-- walk_to: hand an A* path (computed Python-side via request_path/get_path) to
-- the tower_defense scenario's on_tick walker. The walker advances the agent
-- character along this polyline over subsequent ticks while the game runs, so
-- movement is asynchronous and costs real game-time (biters keep attacking).
--
-- coords_str: "x1,y1;x2,y2;..." in world coordinates.

storage.actions.walk_to = function(player_index, coords_str, target_anchor, speed)
    local path = {}
    for pair in string.gmatch(coords_str or "", "[^;]+") do
        local x, y = string.match(pair, "([^,]+),([^,]+)")
        if x and y then
            path[#path + 1] = {x = tonumber(x), y = tonumber(y)}
        end
    end
    storage.td_walk = {
        active = (#path > 0),
        path = path,
        idx = 1,
        target_anchor = target_anchor or -1,
        speed = speed or 0.2,
    }
    return #path
end
