-- Read-only, Kirby-centered terrain probe.
-- Currently limited to the four-block room we have been testing.

local DOMAIN = "System Bus"
local TILE_SIZE = 16
local BLOCK_COLUMNS = 16
local BLOCK_ROWS = 12
local BLOCK_BYTES = 0xC0
local GRID_RADIUS = 4  -- Four cells each way gives a 9x9 grid.

local function read8(address)
    -- Read one byte without changing the game.
    return memory.read_u8(address, DOMAIN)
end

local function read16(low_address, high_address)
    -- Combine separate low and high bytes.
    return read8(low_address) + 256 * read8(high_address)
end

-- Keep every reading in this snapshot on the same game frame.
client.pause()

local room = read16(0x055E, 0x055F)
local kirby_x = read16(0x0083, 0x0095)
local kirby_y = read16(0x00B9, 0x00CB)
local map_width = read8(0x67EE)
local map_height = read8(0x67EF)

-- Print the header so we can check the layout interpretation in room 44.
local header = {}
for offset = 0, 15 do
    header[#header + 1] = string.format("%02X", read8(0x67EE + offset))
end

console.log(string.format(
    "Room=%d | Candidate layout=%d blocks wide, %d high",
    room, map_width, map_height
))
console.log("Header: " .. table.concat(header, " "))

-- This offset matched room 43; we are now testing it in room 44.
local ORDER_BASE = 0x67F6
local MAP_DATA_BASE = 0x68C8
local block_count = map_width * map_height

-- Reject dimensions that would read beyond the candidate order table.
assert(
    map_width > 0
        and map_height > 0
        and block_count <= MAP_DATA_BASE - ORDER_BASE,
    "Unexpected map dimensions; inspect the printed header."
)

-- A room position selects a stored block.
-- Repeated block IDs are allowed: several positions may reuse one block.
local block_order = {}
for index = 0, block_count - 1 do
    local block_id = read8(ORDER_BASE + index)

    -- There are sixteen stored blocks, numbered 00 through 0F.
    assert(
        block_id < 16,
        string.format(
            "Unexpected block ID %02X at order index %d.",
            block_id, index
        )
    )

    block_order[index] = block_id
end

-- Print the candidate block arrangement in rows.
console.log("Candidate block order:")
for block_y = 0, map_height - 1 do
    local row = {}

    for block_x = 0, map_width - 1 do
        local index = block_y * map_width + block_x
        row[#row + 1] = string.format("%02X", block_order[index])
    end

    console.log(table.concat(row, " "))
end

-- Room layouts beyond our tested examples still need visual verification.
console.log(
    "Diagnostic lookup: verify terrain alignment in each new room."
)

-- Our vertical block spacing has not yet been tested in a tall room.
if map_height > 1 then
    console.log(
        "Tall room: vertical block mapping is not yet verified."
    )
end

local function attribute_at(tile_x, tile_y)
    -- World tile coordinates can fall outside the room near its edges.
    -- Return nil for those cells; outside does not mean empty.
    if tile_x < 0 or tile_y < 0
        or tile_x >= map_width * BLOCK_COLUMNS
        or tile_y >= map_height * BLOCK_ROWS then
        return nil
    end

    -- Find the map block containing this cell.
    local block_x = math.floor(tile_x / BLOCK_COLUMNS)
    local block_y = math.floor(tile_y / BLOCK_ROWS)
    local order_index = block_y * map_width + block_x
    local stored_block = block_order[order_index]

    -- Find the cell within that stored block.
    local local_x = tile_x % BLOCK_COLUMNS
    local local_y = tile_y % BLOCK_ROWS
    local address = 0x68C8
        + stored_block * BLOCK_BYTES
        + local_y * BLOCK_COLUMNS
        + local_x

    -- The map stores a tile ID; its attribute is in a separate lookup table.
    local tile_id = read8(address)
    return read8(0x7F00 + tile_id)
end

-- Center on the cell containing Kirby's reported reference position.
-- This is not a measurement of his whole collision box.
local center_x = math.floor(kirby_x / TILE_SIZE)
local center_y = math.floor(kirby_y / TILE_SIZE)

console.log(string.format(
    "Room=%d | Kirby X=%d Y=%d | Center cell=(%d,%d)",
    room, kirby_x, kirby_y, center_x, center_y
))
console.log("Columns and rows run from -4 through +4 relative to Kirby.")
console.log("[XX] = center cell | -- = outside room")
-- Keep every attribute distinct, including possible platforms and water.
console.log("Raw attributes: do not assume unfamiliar values are solid or empty.")

for dy = -GRID_RADIUS, GRID_RADIUS do
    local cells = {}

    for dx = -GRID_RADIUS, GRID_RADIUS do
        local attribute = attribute_at(center_x + dx, center_y + dy)
        local text = "--"

        if attribute ~= nil then
            text = string.format("%02X", attribute)
        end

        -- Brackets mark Kirby's cell without hiding its terrain attribute.
        if dx == 0 and dy == 0 then
            text = "[" .. text .. "]"
        else
            text = " " .. text .. " "
        end

        cells[#cells + 1] = text
    end

    console.log(string.format(
        "%+d: %s", dy, table.concat(cells, " ")
    ))
end

console.log("Snapshot complete. Game remains paused.")