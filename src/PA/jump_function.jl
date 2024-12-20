export glaubitz_jump1, glaubitz_jump2, step_function

function glaubitz_jump1(x)
    if x >= -1 && x <= -0.5
        return 5 / 3 * (1 - x^2)^2
    elseif x > -0.5 && x <= 0.5
        return cos(2 * π * x)
    elseif x > 0.5 && x <= 1
        return (1 - x^2)^4
    end
end

function glaubitz_jump2(x)
    if x >= -1 && x <= -0.5
        return 4 * (x + 0.75)
    elseif x > -0.5 && x <= 0.0
        return -1.0
    elseif x > 0 && x <= 0.5
        return cos(3 / 2 * π * x)
    elseif x > 0.5 && x <= 1
        return cos(7 / 2 * π * x)
    end
end

function step_function(x)
    if x >= -1 && x <= 0
        return 0.0
    elseif x > 0 && x <= 1
        return 1.0
    end
end
