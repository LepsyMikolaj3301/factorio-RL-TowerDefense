-- threat_view: cluster live enemy units into GROUPS (swarms) and list NESTS
-- (unit-spawners), so the model perceives a marching swarm as one threat object
-- distinct from a spawn point.
--
-- Returns a compact string:  "G:cx,cy,count,spread,vx,vy;...|N:x,y,health;..."
-- All values are in WORLD coordinates / raw units; the Python env converts to
-- radar-relative, normalized features. Groups are sorted by count desc and nests
-- by distance to center asc, then capped to max_groups / max_nests.

storage.actions.threat_view = function(player_index, center_x, center_y, radius,
                                       cell, charted_only, max_groups, max_nests)
    local surface = game.surfaces[1]
    local force = game.forces["player"]
    cell = cell or 6
    max_groups = max_groups or 12
    max_nests = max_nests or 8

    local function is_charted(x, y)
        if not charted_only then return true end
        return force.is_chunk_charted(surface, {math.floor(x / 32), math.floor(y / 32)})
    end

    -- 1) Bucket live enemy units into coarse grid cells around the base.
    local area = {{center_x - radius, center_y - radius}, {center_x + radius, center_y + radius}}
    local cells = {}
    local units = surface.find_entities_filtered{area = area, type = "unit", force = "enemy"}
    for _, u in pairs(units) do
        if u.valid then
            local x, y = u.position.x, u.position.y
            if is_charted(x, y) then
                local cxk = math.floor(x / cell)
                local cyk = math.floor(y / cell)
                local key = cxk .. ":" .. cyk
                local c = cells[key]
                if c == nil then
                    c = {sx = 0, sy = 0, n = 0, cxk = cxk, cyk = cyk}
                    cells[key] = c
                end
                c.sx = c.sx + x
                c.sy = c.sy + y
                c.n = c.n + 1
            end
        end
    end

    -- 2) Merge adjacent occupied cells (8-neighbourhood connected components).
    local groups = {}
    local visited = {}
    for key, _ in pairs(cells) do
        if not visited[key] then
            local stack = {key}
            visited[key] = true
            local gsx, gsy, gn = 0, 0, 0
            local minx, miny, maxx, maxy = math.huge, math.huge, -math.huge, -math.huge
            while #stack > 0 do
                local k = table.remove(stack)
                local cc = cells[k]
                gsx = gsx + cc.sx
                gsy = gsy + cc.sy
                gn = gn + cc.n
                local mx = cc.sx / cc.n
                local my = cc.sy / cc.n
                if mx < minx then minx = mx end
                if my < miny then miny = my end
                if mx > maxx then maxx = mx end
                if my > maxy then maxy = my end
                for dx = -1, 1 do
                    for dy = -1, 1 do
                        if not (dx == 0 and dy == 0) then
                            local nk = (cc.cxk + dx) .. ":" .. (cc.cyk + dy)
                            if cells[nk] and not visited[nk] then
                                visited[nk] = true
                                stack[#stack + 1] = nk
                            end
                        end
                    end
                end
            end
            local spread = math.sqrt((maxx - minx) ^ 2 + (maxy - miny) ^ 2) / 2
            groups[#groups + 1] = {cx = gsx / gn, cy = gsy / gn, n = gn, spread = spread}
        end
    end

    -- 3) Cheap velocity via nearest-centroid match against the previous frame.
    local prev = storage.td_threat_prev or {tick = 0, groups = {}}
    local dt = game.tick - (prev.tick or 0)
    if dt < 1 then dt = 1 end
    for _, g in ipairs(groups) do
        local best, bestd = nil, 16 * 16  -- match within 16 tiles
        for _, p in ipairs(prev.groups) do
            local d = (g.cx - p.cx) ^ 2 + (g.cy - p.cy) ^ 2
            if d < bestd then bestd = d; best = p end
        end
        if best then
            g.vx = (g.cx - best.cx) / dt
            g.vy = (g.cy - best.cy) / dt
        else
            g.vx = 0; g.vy = 0
        end
    end
    local saved = {}
    for _, g in ipairs(groups) do saved[#saved + 1] = {cx = g.cx, cy = g.cy} end
    storage.td_threat_prev = {tick = game.tick, groups = saved}

    table.sort(groups, function(a, b) return a.n > b.n end)

    -- 4) Nests: every enemy unit-spawner (searched surface-wide so we catch ones
    --    just outside the grid), nearest first.
    local nests = {}
    local spawners = surface.find_entities_filtered{type = "unit-spawner", force = "enemy"}
    for _, s in pairs(spawners) do
        if s.valid and is_charted(s.position.x, s.position.y) then
            nests[#nests + 1] = {
                x = s.position.x, y = s.position.y, hp = s.health or 0,
                d = (s.position.x - center_x) ^ 2 + (s.position.y - center_y) ^ 2,
            }
        end
    end
    table.sort(nests, function(a, b) return a.d < b.d end)

    -- 5) Encode.
    local g_parts = {}
    for i = 1, math.min(#groups, max_groups) do
        local g = groups[i]
        g_parts[#g_parts + 1] = string.format(
            "%.2f,%.2f,%d,%.2f,%.3f,%.3f", g.cx, g.cy, g.n, g.spread, g.vx, g.vy)
    end
    local n_parts = {}
    for i = 1, math.min(#nests, max_nests) do
        local nn = nests[i]
        n_parts[#n_parts + 1] = string.format("%.2f,%.2f,%.1f", nn.x, nn.y, nn.hp)
    end

    return "G:" .. table.concat(g_parts, ";") .. "|N:" .. table.concat(n_parts, ";")
end
