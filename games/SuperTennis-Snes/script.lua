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

---Calculates SuperTennis reward for step
---@return number
function calculate_reward()
    local reward = 0
    if data.player_points > playerScore then
        local delta = data.player_points - playerScore
        if rewardPlayerPoints then
            reward = reward + delta
        end
        playerScore = data.player_points
        print("Awarding a player point")
    end

    if data.opponent_points > opponentScore then
        local delta = data.opponent_points - opponentScore
        if rewardOpponentPoints then
            reward = reward - delta
        end
        opponentScore = data.opponent_points
        print("Substracting an opponent point")
    end

    if rewardBallReturns then
        reward = reward + compute_return_reward(data.total_games, data.total_point_returns)
    end
    if rewardBallProximity then
        reward = reward +
            compute_ball_proximity_reward(data.ball_is_bottom_court, data.player_assigned_bottom_court,
                data.player_serving,
                data.ball_x_position, data.ball_y_position, data.player_position_x, data.player_position_y,
                data.frame_count)
    end
    if penalizeStalling then
        reward = reward - compute_stalling_penalty(data.player_serving, data.frame_count)
    end
    if penalizeFaults then
        reward = reward - compute_fault_penalty(data.in_fault, data.total_games)
    end

    return reward
end
