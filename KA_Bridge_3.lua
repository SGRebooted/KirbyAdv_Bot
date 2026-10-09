-- Kirby RAM/control bridge for BizHawk 2.11.1 (NesHawk).
-- Load this instead of the original object logger: only one script should
-- advance frames. The original logger remains available for mapping work.

-- The folder containing KA_Bot.py and KA_Bridge.lua.
-- Forward slashes work on Windows and avoid Lua backslash escape sequences.
local root = "D:/VS_Code_Projects/KirbyAdv_Bot/"

-- KA_Parallel.py creates this worker-specific folder on startup.
-- Both programs exchange their command and response files here.
local worker_id = 3
local folder = root .. "bridge_3/"
local domain = "System Bus"
local last_id = nil
local saved = false
local running = true
local transition_seen = false
local buttons = {"Up", "Down", "Left", "Right", "A", "B", "Start", "Select"}

-- Read low/high bytes explicitly because several RAM fields are not adjacent.
local function read8(address)
    return memory.read_u8(address, domain)
end

local function read16(low, high, signed)
    local value = read8(low) + 256 * read8(high)
    if signed and value >= 32768 then value = value - 65536 end
    return value
end

-- These addresses come from your saved Kirby's Adventure RAM Watch file.
-- Velocity is kept in raw signed units; its pixel scale is not assumed here.
local function player_json()
    return string.format(
        '{"hp":%d,"lives":%d,"ability":%d,"mouth_full":%d,"sprite_state":%d,' ..
        '"x":%d,"y":%d,"vx":%d,"vy":%d,"level":%d,"stage":%d,"room":%d}',
        -- Include the sprite state observed during Warp Star travel.
        read8(0x597), read8(0x599), read8(0x5E3), read8(0x5E0),
        read8(0x62C7),
        read16(0x83, 0x95), read16(0xB9, 0xCB),
        read16(0x5B9, 0x5BA, true), read16(0x5BD, 0x5BE, true),
        read8(0x558), read8(0x559), read16(0x55E, 0x55F))
end

-- Preserve all nine slots and all scratch variables, including empty slots.
-- Scratch variables may be stale; do not use them as confirmed item identities.
local function objects_json()
    local objects = {}
    for slot = 0x09, 0x11 do
        local vars = {}
        for index = 0, 7 do
            vars[#vars + 1] = tostring(read8(0x61CA + index * 0x12 + slot))
        end
        objects[#objects + 1] = string.format(
            '{"slot":%d,"script":%d,"x":%d,"y":%d,"vx":%d,"vy":%d,"pose":%d,"vars":[%s]}',
            slot, read8(0x6128 + slot),
            read16(0x82 + slot, 0x94 + slot), read16(0xB8 + slot, 0xCA + slot),
            read16(0x60E0 + slot, 0x60F2 + slot, true),
            read16(0x6104 + slot, 0x6116 + slot, true),
            read8(0x62C6 + slot), table.concat(vars, ","))
    end
    return "[" .. table.concat(objects, ",") .. "]"
end

local function terrain_json()
    -- During observed loading/travel states, map memory may be changing.
    if read8(0x62C7) == 255 then
        return '{"status":"transition","cells":null}'
    end

    local width = read8(0x67EE)
    local height = read8(0x67EF)
    local order_base = 0x67F6
    local data_base = 0x68C8
    local count = width * height

    -- Reject dimensions that would run beyond the candidate order table.
    if width == 0 or height == 0 or count > data_base - order_base then
        return '{"status":"invalid_dimensions","cells":null}'
    end

    -- Our world-to-block mapping has only been checked in horizontal rooms.
    -- Mark tall rooms unavailable until their vertical spacing is verified.
    if height ~= 1 then
        return '{"status":"vertical_layout_unverified","cells":null}'
    end

    -- Read each room position's stored block ID.
    -- Repeated IDs are allowed; only sixteen different blocks are stored.
    local order = {}
    for index = 0, count - 1 do
        local block_id = read8(order_base + index)
        if block_id >= 16 then
            return '{"status":"invalid_block_order","cells":null}'
        end
        order[index] = block_id
    end

    -- Locate the terrain cell containing Kirby's reference position.
    local center_x = math.floor(read16(0x83, 0x95) / 16)
    local center_y = math.floor(read16(0xB9, 0xCB) / 16)
    local cells = {}

    -- Store 81 attributes in row order: top-left through bottom-right.
    for dy = -4, 4 do
        for dx = -4, 4 do
            local tile_x = center_x + dx
            local tile_y = center_y + dy

            -- -1 means outside the room, distinctly different from FF air.
            local attribute = -1

            if tile_x >= 0 and tile_x < width * 16
                and tile_y >= 0 and tile_y < height * 12 then

                -- Find the stored block, then the tile inside that block.
                local block_x = math.floor(tile_x / 16)
                local block_y = math.floor(tile_y / 12)
                local block_id = order[block_y * width + block_x]

                local address = data_base
                    + block_id * 0xC0
                    + (tile_y % 12) * 16
                    + (tile_x % 16)

                local tile_id = read8(address)
                attribute = read8(0x7F00 + tile_id)
            end

            -- Preserve raw attributes, including ones we haven't identified.
            cells[#cells + 1] = tostring(attribute)
        end
    end

    -- "candidate" means the structure passed checks, not that every room
    -- or attribute has been experimentally verified.
    return string.format(
        '{"status":"candidate","width":%d,"height":%d,' ..
        '"center_x":%d,"center_y":%d,"cells":[%s]}',
        width, height, center_x, center_y, table.concat(cells, ",")
    )
end

-- All error strings below are fixed text, so they contain no JSON escapes.
local function respond(id, error_message)
    local text
    if error_message then
        text = '{"id":"' .. id .. '","error":"' .. error_message .. '"}'
    else
                -- Attach terrain to the same snapshot as Kirby and the objects.
        text = '{"id":"' .. id .. '","frame":' .. emu.framecount() ..
            ',"kirby":' .. player_json() ..
            ',"objects":' .. objects_json() ..
            ',"terrain":' .. terrain_json() ..
            ',"transition_seen":' .. tostring(transition_seen) .. '}'
    end
    -- Identify this worker without changing the RAM snapshot format.
    text = text:sub(1, -2) .. ',"worker_id":' .. worker_id .. '}'
    local file = assert(io.open(folder .. "response.json", "w"), "Start KA_Parallel.py first")
    file:write(text)
    file:close()
end

-- Explicitly set every button, preventing the last action from staying held.
local function released_buttons()
    local result = {}
    for _, name in ipairs(buttons) do result[name] = false end
    return result
end

-- A command executes once even though its file remains on disk afterward.
local function process(message)
    local id, command, frame_text, keys = message:match("^(%x+)|(%a+)|(%d+)|(.*)$")
    if not id or id == last_id then return end
    last_id = id
    transition_seen = false
    local frames = tonumber(frame_text)
    local held = released_buttons()
    for key in keys:gmatch("[^,]+") do
        if held[key] == nil then respond(id, "Unknown controller button"); return end
        held[key] = true
    end
    if command == "step" then
        if frames < 1 or frames > 60 then respond(id, "Frames must be 1 to 60"); return end
        -- Allow the emulator to run while we execute this command.
        client.unpause()

        -- Apply the chosen buttons before advancing each frame.
        -- Check before and after every frame so short transitions are noticed.
        for _ = 1, frames do
            if read8(0x62C7) == 255 then
                transition_seen = true
            end

            -- Once a transition begins, keep the rest of this command neutral.
            -- This prevents an old input from carrying into the destination.
            if transition_seen then
                joypad.set(released_buttons(), 1)
            else
                joypad.set(held, 1)
            end

            emu.frameadvance()

            -- Include transitions that begin on the command's final frame.
            if read8(0x62C7) == 255 then
                transition_seen = true
            end
        end

        -- Freeze the game again while Python chooses the next action.
        client.pause()
    elseif command == "save" then
        -- Use a dedicated file so existing numbered savestate slots are untouched.
        savestate.save(folder .. "start.State")
        saved = true
    elseif command == "reset" then
        if not saved then respond(id, "Use save first in this Lua session"); return end
        savestate.load(folder .. "start.State")
    elseif command == "stop" then
        running = false
    elseif command ~= "observe" then
        respond(id, "Unknown command"); return
    end
    joypad.set(released_buttons(), 1)
    client.pause()
    respond(id)
end

-- Discard a stale command from a previous run, especially an old movement input.
local old = io.open(folder .. "command.txt", "r")
if old then last_id = old:read("*a"):match("^(%x+)|"); old:close() end
client.pause()
console.log("Kirby worker 1 ready. Use read 1 in KA_Parallel.py. Only this bridge should run in this window.")
while running do
    local file = io.open(folder .. "command.txt", "r")
    if file then
        local message = file:read("*a")
        file:close()
        process(message)
    end
    -- Yield keeps the UI responsive without advancing uncontrolled game frames.
    emu.yield()
end
joypad.set(released_buttons(), 1)
console.log("Kirby bridge stopped. The emulator is paused.")
