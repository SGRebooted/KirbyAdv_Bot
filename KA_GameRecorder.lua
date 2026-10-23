-- Read-only game research. Run by itself in a manual BizHawk instance.
-- Edit these settings for each experiment. The diagnostics folder must exist.
local ROOT = "C:/VSCode_Projects_SSD/KA_Bot_SSD"
local LABEL = "Kirby_i_frame_test"
local STRIDE = 1               -- 1 for attacks/jumps; 4 for broad scouting
local MAX_FRAMES = 1800        -- stops recording after this many frame callbacks
local FULL_EVERY = 120         -- full snapshot every 120 samples; deltas otherwise

assert(STRIDE == 1 or STRIDE == 4, "STRIDE must be 1 or 4")
assert(MAX_FRAMES >= 0 and MAX_FRAMES == math.floor(MAX_FRAMES))
assert(FULL_EVERY > 0 and FULL_EVERY == math.floor(FULL_EVERY))
assert(LABEL:match("^[%w_-]+$"), "Use letters, digits, underscore or dash for LABEL")
-- Only RAM ranges: do not sweep PPU/APU/controller I/O through System Bus.
local ranges = {{0x0000, 0x07FF}, {0x6000, 0x7FFF}}
local domain = "System Bus"
local prefix = ROOT .. "/diagnostics/game_" .. os.date("!%Y%m%dT%H%M%SZ") .. "_" .. LABEL
local path, file
for suffix = 1, 10000 do
    local candidate = prefix .. "_" .. suffix .. ".jsonl"
    local existing = io.open(candidate, "rb")
    if existing then existing:close() else
        path = candidate
        file = assert(io.open(path, "w"), "Create the diagnostics folder first")
        break
    end
end
assert(file, "Could not allocate recording name")
local stopped, count, ticks, segment = false, 0, 0, 0
local last_frame, previous = nil, nil
local callbacks = {}
local function write(text)
    assert(file:write(text .. "\n"))
    assert(file:flush())
end
local function finish(reason)
    if stopped then return end
    stopped = true
    local ok, err = pcall(function()
        write(string.format('{"type":"end","reason":"%s","samples":%d,"elapsed_frames":%d}', reason, count, ticks))
    end)
    file:close()
    for _, id in ipairs(callbacks) do event.unregisterbyid(id) end
    console.log("Game recorder stopped: " .. reason .. " | " .. path)
    if not ok then console.log("Recording write failed: " .. tostring(err)) end
end
local function discontinuity(reason)
    segment = segment + 1
    previous = nil
    last_frame = nil
    write(string.format('{"type":"boundary","segment":%d,"reason":"%s","frame":%d}', segment, reason, emu.framecount()))
end
local function sample()
    local frame = emu.framecount()
    if last_frame and frame < last_frame then discontinuity("frame_rewind") end
    local current, changes, blocks = {}, {}, {}
    local full = previous == nil or count % FULL_EVERY == 0
    for _, range in ipairs(ranges) do
        local hex = {}
        for addr = range[1], range[2] do
            local value = memory.read_u8(addr, domain)
            current[addr] = value
            if full then hex[#hex + 1] = string.format("%02X", value)
            elseif previous[addr] ~= value then
                changes[#changes + 1] = string.format('"%04X":%d', addr, value)
            end
        end
        if full then blocks[#blocks + 1] = string.format('"%04X":"%s"', range[1], table.concat(hex)) end
    end
    local keys = joypad.get(1)
    local pressed = {}
    for _, key in ipairs({"Up", "Down", "Left", "Right", "A", "B", "Start", "Select"}) do
        if keys[key] then pressed[#pressed + 1] = '"' .. key .. '"' end
    end
    write(string.format('{"type":"sample","sample":%d,"frame":%d,"elapsed_frames":%d,"segment":%d,"buttons":[%s],"%s":{%s}}',
        count, frame, ticks, segment, table.concat(pressed, ","), full and "full" or "delta", table.concat(full and blocks or changes, ",")))
    previous, last_frame = current, frame
    count = count + 1
end
local function guarded(fn)
    if stopped then return end
    local ok, err = pcall(fn)
    if not ok then
        console.log("Game recorder error: " .. tostring(err))
        finish("error")
    end
end
write(string.format('{"type":"header","schema":"ka-game-ram-v1","label":"%s","utc":"%s","domain":"System Bus","stride":%d,"max_frames":%d,"ranges":[[0,2047],[24576,32767]],"input_source":"joypad.get(1) at sample time"}', LABEL, os.date("!%Y-%m-%dT%H:%M:%SZ"), STRIDE, MAX_FRAMES))
callbacks[#callbacks + 1] = event.onframeend(function()
    guarded(function()
        ticks = ticks + 1
        if ticks % STRIDE == 0 then sample() end
        if MAX_FRAMES > 0 and ticks >= MAX_FRAMES then finish("frame_limit") end
    end)
end, "KA research frame")
callbacks[#callbacks + 1] = event.onloadstate(function()
    guarded(function() discontinuity("savestate_load"); sample() end)
end, "KA research load")
event.onexit(function() finish("script_stopped") end, "KA research exit")
console.log("Recording: " .. path)
console.log("Manual play: unpause or frame-step normally. Stop this Lua script to finish early.")
guarded(sample)
-- Yield leaves pause, speed, controller input and frame stepping to the user.
while not stopped do emu.yield() end
