-- read_td_events: return the accumulated tower-defense reward event counters as
-- a compact CSV string and RESET them (read-and-clear). The counters are filled
-- by the on_entity_died handler in the tower_defense scenario control.lua.
-- Order: kills,turrets_lost,walls_lost,buildings_lost,radar_lost,char_died

storage.actions.read_td_events = function(player_index)
    local ev = storage.td_events
    if ev == nil then
        ev = {kills = 0, turrets_lost = 0, walls_lost = 0,
              buildings_lost = 0, radar_lost = 0, char_died = 0}
    end
    local out = string.format("%d,%d,%d,%d,%d,%d",
        ev.kills or 0, ev.turrets_lost or 0, ev.walls_lost or 0,
        ev.buildings_lost or 0, ev.radar_lost or 0, ev.char_died or 0)
    -- Reset for the next decision window.
    storage.td_events = {kills = 0, turrets_lost = 0, walls_lost = 0,
                         buildings_lost = 0, radar_lost = 0, char_died = 0}
    return out
end
