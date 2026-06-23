-- walk_to: hand an A* path (computed Python-side via request_path/get_path) to
-- the FLE tick dispatcher. The dispatcher advances the agent character along
-- this polyline over subsequent ticks while the game runs, so movement is
-- asynchronous and costs real game-time (biters keep attacking).
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
        player_index = player_index or 1,
        target_anchor = target_anchor or -1,
        speed = speed or 0.2,
    }
    return #path
end

storage.actions.update_td_walk = function(event)
    if storage.td_walk == nil then
        storage.td_walk = {
            active = false,
            path = {},
            idx = 1,
            player_index = 1,
            target_anchor = -1,
            speed = 0.2,
        }
    end

    local walk = storage.td_walk
    if not walk.active then return end

    local player_index = walk.player_index or 1
    local char = storage.agent_characters and storage.agent_characters[player_index]
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
end
