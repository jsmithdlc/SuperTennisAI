--[[
Secondary rewards .lua module to add intermediate rewards for
making agent learning easier
]] --

-- Return ball secondary reward variables
ReturnReward = 0.5      -- reward per returned ball
CurrentTotalReturns = 0 -- tracks balls returned by player

-- Proximity to ball secondary reward variables
ProximityLevel1RadiusPix = 30     -- first (closest) level of proximity distance in pixels
ProximityLevel1Reward = 0.2       -- first (closest) level of proximity reward
ProximityLevel2RadiusPix = 60     -- second level of proximity distance in pixels
ProximityLevel2Reward = 0.1       -- second level of proximity reward
ProximityLevel3RadiusPix = 80     -- third (furthest) level of proximity distance in pixels
ProximityLevel3Reward = 0.05      -- third (furthest) level of proximity reward
ProximityRewardFrameCooldown = 15 -- # frame cooldown before re-awarding
LastProximityRewardFrame = 0      -- Last frame index in which proximity reward was applied

--- Checks if player is returning a ball
---@param totalGames integer game variable that tracks total games played
---@param totalPointReturns integer game variable that tracks total ball returns
---@return boolean
local function check_if_player_returns_ball(totalGames, totalPointReturns)
    if totalGames % 2 == 0 and totalPointReturns % 2 == 0 then
        return true
    elseif totalGames % 2 ~= 0 and totalPointReturns % 2 ~= 0 then
        return true
    else
        return false
    end
end

---Computes secondary reward for ball returns
---@param totalGames integer game variable that tracks total games played
---@param totalPointReturns integer game variable that tracks total ball returns
---@return number
function compute_return_reward(totalGames, totalPointReturns)
    local is_player_return = check_if_player_returns_ball(totalGames, totalPointReturns)
    local delta_returns = totalPointReturns - CurrentTotalReturns
    CurrentTotalReturns = totalPointReturns
    if delta_returns > 0 and is_player_return then
        return ReturnReward
    end
    return 0
end

--- Calculates cartesian distance from player to ball
---@param ballX integer x-position of ball, in pixels
---@param ballY integer y-position of ball, in pixels
---@param playerX integer x-position of player, in pixels
---@param playerY integer y-position of player, in pixels
---@return number
local function calculate_distance_to_ball(ballX, ballY, playerX, playerY)
    local deltaX2 = (ballX - playerX) ^ 2
    local deltaY2 = (ballY - playerY) ^ 2
    return math.sqrt(deltaX2 + deltaY2)
end

---Computes proximity to ball secondary reward
---@param ballIsBottomCourt integer game variable that tracks if ball is at bottom court when set to 1
---@param playerAssignedBottomCourt integer game variable that tracks if player was assigned bottom court in this game when set to 1
---@param playerServingValue integer game variable that tracks if player is serving
---@param ballX integer x-position of ball, in pixels
---@param ballY integer y-position of ball, in pixels
---@param playerX integer x-position of player, in pixels
---@param playerY integer y-position of player, in pixels
---@param frameCount integer total game frame count
---@return number
function compute_ball_proximity_reward(ballIsBottomCourt, playerAssignedBottomCourt, playerServingValue, ballX, ballY,
                                       playerX,
                                       playerY, frameCount)
    local distanceToBallPix = calculate_distance_to_ball(ballX, ballY, playerX, playerY)

    if (distanceToBallPix > ProximityLevel3RadiusPix) or (ballIsBottomCourt ~= playerAssignedBottomCourt) or ((frameCount - LastProximityRewardFrame) < ProximityRewardFrameCooldown) or (playerServingValue == 1) or (playerServingValue == 17) then
        return 0
    end
    LastProximityRewardFrame = frameCount
    if distanceToBallPix <= ProximityLevel1RadiusPix then
        return ProximityLevel1Reward
    elseif distanceToBallPix > ProximityLevel1RadiusPix and distanceToBallPix <= ProximityLevel2RadiusPix then
        return ProximityLevel2Reward
    else
        return ProximityLevel3Reward
    end
end
