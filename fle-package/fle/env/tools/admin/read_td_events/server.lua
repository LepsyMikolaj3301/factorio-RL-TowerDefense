-- read_td_events: return the accumulated tower-defense reward event counters as
-- a compact CSV string and RESET them (read-and-clear). The counters are filled
-- by fle/env/mods/td_events.lua, which is runtime-loaded so it works even when
-- a prebuilt save contains stale scenario code.
-- Order:
-- kills,turrets_lost,walls_lost,buildings_lost,boilers_lost,radar_lost,char_died,
-- wall_n,wall_e,wall_s,wall_w,turret_n,turret_e,turret_s,turret_w

storage.actions.read_td_events = function(player_index)
    local ev = storage.td_events
    if ev == nil then
        if storage.actions and storage.actions.td_zero_events then
            ev = storage.actions.td_zero_events()
        else
            ev = {kills = 0, turrets_lost = 0, walls_lost = 0,
                  buildings_lost = 0, boilers_lost = 0,
                  radar_lost = 0, char_died = 0,
                  wall_n = 0, wall_e = 0, wall_s = 0, wall_w = 0,
                  turret_n = 0, turret_e = 0, turret_s = 0, turret_w = 0}
        end
    end
    local out = string.format("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
        ev.kills or 0, ev.turrets_lost or 0, ev.walls_lost or 0,
        ev.buildings_lost or 0, ev.boilers_lost or 0,
        ev.radar_lost or 0, ev.char_died or 0,
        ev.wall_n or 0, ev.wall_e or 0, ev.wall_s or 0, ev.wall_w or 0,
        ev.turret_n or 0, ev.turret_e or 0, ev.turret_s or 0, ev.turret_w or 0)
    -- Reset for the next decision window.
    if storage.actions and storage.actions.td_zero_events then
        storage.td_events = storage.actions.td_zero_events()
    else
        storage.td_events = {kills = 0, turrets_lost = 0, walls_lost = 0,
                             buildings_lost = 0, boilers_lost = 0,
                             radar_lost = 0, char_died = 0,
                             wall_n = 0, wall_e = 0, wall_s = 0, wall_w = 0,
                             turret_n = 0, turret_e = 0, turret_s = 0, turret_w = 0}
    end
    return out
end
