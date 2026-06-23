-- Runtime tower-defense reward events.
--
-- Prebuilt saves may carry an old or unrelated scenario control.lua, while FLE
-- tools are injected into the loaded game at runtime. Register TD death events
-- here so loss/kills counters are available regardless of the save's script.

if not storage.actions then
    storage.actions = {}
end

local function td_zero_events()
    return {
        kills = 0,
        turrets_lost = 0,
        walls_lost = 0,
        buildings_lost = 0,
        boilers_lost = 0,
        radar_lost = 0,
        char_died = 0,
        wall_n = 0,
        wall_e = 0,
        wall_s = 0,
        wall_w = 0,
        turret_n = 0,
        turret_e = 0,
        turret_s = 0,
        turret_w = 0,
    }
end

storage.actions.td_zero_events = td_zero_events

local function td_events()
    if storage.td_events == nil then
        storage.td_events = td_zero_events()
    end
    return storage.td_events
end

storage.actions.td_ensure_events = td_events

local function td_radar_position(surface)
    local radar = surface.find_entities_filtered{name = "radar"}[1]
    if radar and radar.valid then
        return radar.position
    end
    return {x = 0, y = 0}
end

local function td_direction_key(entity)
    local pos = entity.position
    local center = td_radar_position(entity.surface)
    local dx = pos.x - center.x
    local dy = pos.y - center.y
    if math.abs(dx) >= math.abs(dy) then
        if dx >= 0 then return "e" end
        return "w"
    end
    if dy >= 0 then return "s" end
    return "n"
end

script.on_event(defines.events.on_entity_died, function(event)
    local ent = event.entity
    if not (ent and ent.valid) then return end

    local ev = td_events()
    local fname = ent.force and ent.force.name or ""
    local etype = ent.type
    local ename = ent.name

    if fname == "enemy" then
        if etype == "unit" then
            ev.kills = (ev.kills or 0) + 1
        end
        return
    end

    if fname ~= "player" then return end

    if etype == "ammo-turret" or etype == "electric-turret"
        or etype == "fluid-turret" or ename == "gun-turret" then
        ev.turrets_lost = (ev.turrets_lost or 0) + 1
        local key = "turret_" .. td_direction_key(ent)
        ev[key] = (ev[key] or 0) + 1
    elseif etype == "wall" or etype == "gate" then
        ev.walls_lost = (ev.walls_lost or 0) + 1
        local key = "wall_" .. td_direction_key(ent)
        ev[key] = (ev[key] or 0) + 1
    elseif etype == "radar" or ename == "radar" then
        ev.radar_lost = (ev.radar_lost or 0) + 1
    elseif ename == "boiler" then
        ev.boilers_lost = (ev.boilers_lost or 0) + 1
        ev.buildings_lost = (ev.buildings_lost or 0) + 1
    elseif etype == "character" then
        ev.char_died = 1
    else
        ev.buildings_lost = (ev.buildings_lost or 0) + 1
    end
end)
