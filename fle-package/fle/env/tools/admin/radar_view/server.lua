-- radar_view: returns a base64-encoded uint8[C,H,W] symbolic grid
-- Channels: 0=empty, 1=wall, 2=turret, 3=ammo_pct, 4=biter, 5=spitter, 6=spawner, 7=character

storage.actions.radar_view = function(player_index, center_x, center_y, radius, cell_size, charted_only)
    local surface = game.surfaces[1]
    local force = game.forces["player"]
    local grid_size = math.floor(2 * radius / cell_size)
    local num_channels = 8

    -- Pre-allocate flat grid (C * H * W)
    local grid = {}
    local total = num_channels * grid_size * grid_size
    for i = 1, total do
        grid[i] = 0
    end

    -- Helper: set grid value at (channel, row, col) - 0-indexed channel/row/col
    local function set_grid(ch, row, col, val)
        if row >= 0 and row < grid_size and col >= 0 and col < grid_size then
            local idx = ch * grid_size * grid_size + row * grid_size + col + 1
            grid[idx] = math.min(255, math.max(0, math.floor(val)))
        end
    end

    -- Helper: ACCUMULATE a value at (channel, row, col), clipped to 255. Used for
    -- the biter/spitter channels so a dense swarm reads as a high-intensity blob
    -- (distinguishable from scattered stragglers, and never confused with the
    -- single-point nest channel).
    local function add_grid(ch, row, col, val)
        if row >= 0 and row < grid_size and col >= 0 and col < grid_size then
            local idx = ch * grid_size * grid_size + row * grid_size + col + 1
            grid[idx] = math.min(255, grid[idx] + val)
        end
    end
    -- Per-unit intensity for the count-valued enemy channels (4 units saturate a cell).
    local UNIT_INTENSITY = 64

    -- Convert world position to grid position
    local function world_to_grid(wx, wy)
        local col = math.floor((wx - center_x + radius) / cell_size)
        local row = math.floor((wy - center_y + radius) / cell_size)
        return row, col
    end

    local area = {
        {center_x - radius, center_y - radius},
        {center_x + radius, center_y + radius}
    }

    -- Check if position is in a charted chunk
    local function is_charted(x, y)
        if not charted_only then return true end
        local chunk_x = math.floor(x / 32)
        local chunk_y = math.floor(y / 32)
        return force.is_chunk_charted(surface, {chunk_x, chunk_y})
    end

    -- Scan player force entities (walls, turrets, radar, character)
    local player_entities = surface.find_entities_filtered{area = area, force = "player"}
    for _, entity in pairs(player_entities) do
        if entity.valid and is_charted(entity.position.x, entity.position.y) then
            local row, col = world_to_grid(entity.position.x, entity.position.y)
            local etype = entity.type

            if etype == "wall" or etype == "gate" then
                set_grid(1, row, col, 255)
            elseif etype == "ammo-turret" or etype == "electric-turret" or etype == "fluid-turret" then
                set_grid(2, row, col, 255)
                -- Ammo percentage for ammo turrets
                if etype == "ammo-turret" then
                    local inv = entity.get_inventory(defines.inventory.turret_ammo)
                    if inv then
                        local used = 0
                        local cap = #inv
                        for i = 1, #inv do
                            if inv[i].valid_for_read then
                                used = used + 1
                            end
                        end
                        local pct = cap > 0 and (used / cap * 255) or 0
                        set_grid(3, row, col, pct)
                    end
                end
            elseif etype == "character" then
                set_grid(7, row, col, 255)
            end
        end
    end

    -- Scan enemy entities (biters, spitters, spawners)
    local enemy_entities = surface.find_entities_filtered{area = area, force = "enemy"}
    for _, entity in pairs(enemy_entities) do
        if entity.valid and is_charted(entity.position.x, entity.position.y) then
            local row, col = world_to_grid(entity.position.x, entity.position.y)
            local etype = entity.type
            local ename = entity.name

            if etype == "unit" then
                if string.find(ename, "biter") then
                    add_grid(4, row, col, UNIT_INTENSITY)
                elseif string.find(ename, "spitter") then
                    add_grid(5, row, col, UNIT_INTENSITY)
                end
            elseif etype == "unit-spawner" then
                set_grid(6, row, col, 255)
            end
        end
    end

    -- Encode as base64
    local bytes = {}
    for i = 1, total do
        bytes[i] = string.char(grid[i])
    end
    local raw = table.concat(bytes)

    -- Simple base64 encoding
    local b64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    local result = {}
    local n = #raw
    for i = 1, n, 3 do
        local a = string.byte(raw, i) or 0
        local b_val = (i + 1 <= n) and string.byte(raw, i + 1) or 0
        local c_val = (i + 2 <= n) and string.byte(raw, i + 2) or 0

        local bits = a * 65536 + b_val * 256 + c_val

        local o1 = math.floor(bits / 262144) % 64
        local o2 = math.floor(bits / 4096) % 64
        local o3 = math.floor(bits / 64) % 64
        local o4 = bits % 64

        result[#result + 1] = string.sub(b64chars, o1 + 1, o1 + 1)
        result[#result + 1] = string.sub(b64chars, o2 + 1, o2 + 1)
        if i + 1 <= n then
            result[#result + 1] = string.sub(b64chars, o3 + 1, o3 + 1)
        else
            result[#result + 1] = "="
        end
        if i + 2 <= n then
            result[#result + 1] = string.sub(b64chars, o4 + 1, o4 + 1)
        else
            result[#result + 1] = "="
        end
    end

    return "b64:" .. table.concat(result)
end
