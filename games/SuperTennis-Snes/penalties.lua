--[[
Penalties .lua module to guide agent away
from unwanted behavior.

]] --

-- Stalling penalty variables
StallPenalty = 0.5          -- penalty magnitude
FrameStartStalling = 0      -- frame id in which player started stalling
FramesTillPenalty = 360     -- # frames until penalty is applied
PlayerMarkedServing = false -- tracks if customer is serving

-- Double Fault penalty variables
FaultPenalty = 1       -- penalty magnitude
PlayerHasFault = false -- tracks if customer is already at fault

function Set(list)
    local set = {}
    for _, value in ipairs(list) do
        set[value] = true
    end
    return set
end

--- Computes stalling penalty
--- @param playerServeValue integer game variable that tracks player serving
---          penalized values are 1 (ball at hand / launched into air) and 17 (dropped to floor)
--- @param frameCount integer game frame count
--- @return number
function compute_stalling_penalty(playerServeValue, frameCount)
    local stallingValues = Set({ 1, 17 })
    if stallingValues[playerServeValue] and not PlayerMarkedServing then
        FrameStartStalling = frameCount
        PlayerMarkedServing = true
        FramesTillPenalty = 360
    elseif stallingValues[playerServeValue] and PlayerMarkedServing then
        local delta_stalling_frames = frameCount - FrameStartStalling
        if delta_stalling_frames >= FramesTillPenalty then
            -- Once player starts stalling, reapply penalty
            -- if recurring
            FramesTillPenalty = 80
            FrameStartStalling = frameCount
            print("Adding penalty for stalling")
            return StallPenalty
        end
    else
        PlayerMarkedServing = false
    end
    return 0
end

--- Computes double fault penalty
--- @param inFault integer game variable that tracks if player has faulted
--- @param totalGames integer game variable that tracks the total number of games played
function compute_fault_penalty(inFault, totalGames)
    if inFault == 1 and totalGames % 2 == 0 and not PlayerHasFault then
        PlayerHasFault = (inFault == 1)
        print("Adding penalty for faulting")
        return FaultPenalty
    end
    PlayerHasFault = (inFault == 1)
    return 0
end
