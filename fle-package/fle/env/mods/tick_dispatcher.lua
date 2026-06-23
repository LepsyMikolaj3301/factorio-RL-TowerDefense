-- Central per-tick dispatcher for FLE-loaded runtime jobs.
--
-- Factorio only keeps one handler for script.on_event(defines.events.on_tick).
-- Loading multiple files that each register on_tick silently replaces earlier
-- handlers. Keep the single registration here and dispatch optional jobs.

if not storage.actions then
    storage.actions = {}
end

local function call_tick_action(name, event)
    local fn = storage.actions and storage.actions[name]
    if not fn then return end

    local ok, err = pcall(fn, event)
    if not ok then
        storage.tick_dispatch_errors = storage.tick_dispatch_errors or {}
        storage.tick_dispatch_errors[name] = tostring(err)
    end
end

script.on_event(defines.events.on_tick, function(event)
    storage.elapsed_ticks = (storage.elapsed_ticks or 0) + 1

    call_tick_action("update_crafting_queue", event)
    call_tick_action("update_td_walk", event)
    call_tick_action("update_alerts", event)
end)
