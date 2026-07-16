--[[
Script that calculates SuperTennis step reward, to use with
stable-retro library
]] --

-- Variables that track scores
local playerScore = 0
local opponentScore = 0

-- Reward / Penalty controls
local rewardPlayerPoints = true
local rewardOpponentPoints = true
local rewardBallReturns = true
local rewardBallProximity = true
local penalizeStalling = true
local penalizeFaults = true

-- logging
local logReward = false

---Calculates SuperTennis reward for step
---@return number
function calculate_reward()
    local reward = 0
    local rewardInfo = { "Reward calculation breakdown:" }
    if data.player_points > playerScore then
        local delta = data.player_points - playerScore
        if rewardPlayerPoints then
            reward = reward + delta
        end
        playerScore = data.player_points
        table.insert(rewardInfo, "  +1.0 reward point for player score")
    end

    if data.opponent_points > opponentScore then
        local delta = data.opponent_points - opponentScore
        if rewardOpponentPoints then
            reward = reward - delta
        end
        opponentScore = data.opponent_points
        table.insert(rewardInfo, "  -1.0 reward point for opponent score")
    end

    if rewardBallReturns then
        local returnReward = compute_return_reward(data.total_games, data.total_point_returns)
        reward = reward + returnReward
        table.insert(rewardInfo, string.format("  +%.1f reward points for ball return!", ReturnReward))
    end
    if rewardBallProximity then
        local proximityReward = compute_ball_proximity_reward(data.ball_is_bottom_court,
            data.player_assigned_bottom_court,
            data.player_serving,
            data.ball_x_position, data.ball_y_position, data.player_position_x, data.player_position_y,
            data.frame_count)
        reward = reward + proximityReward
        table.insert(rewardInfo, string.format("  +%.1f reward points for proximity to ball", proximityReward))
    end
    if penalizeStalling then
        local stallPenalty = compute_stalling_penalty(data.player_serving, data.frame_count)
        reward = reward - stallPenalty
        table.insert(rewardInfo, string.format("  -%.1f reward points for stalling", stallPenalty))
    end
    if penalizeFaults then
        local faultPenalty = compute_fault_penalty(data.in_fault, data.total_games)
        reward = reward - faultPenalty
        table.insert(rewardInfo, string.format("  -%.1f reward points for double fault", faultPenalty))
    end

    if logReward then
        print(table.concat(rewardInfo, "\n"))
    end
    return reward
end
