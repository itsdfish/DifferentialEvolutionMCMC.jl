using StatsBase, LinearAlgebra

mutable struct Model{T1,T2,T3,T4}
    θflag::T1
    θguess::T2
    ϕ::T3
    depth::Int
    focus::T4
    option_selected::Bool
end

function Model(;θflag=.5, θguess=.05, ϕ=1.0, depth=3,
    focus=CartesianIndex(0,0))
    return Model(θflag, θguess, ϕ, depth, focus, false)
end

function flag_cell!(cell)
    cell.flagged = true
end

function flag_mines!(model, game)
    for i in 1:model.depth
        options = search(game, i)
        isempty(options) ? (continue) : nothing
        option = select_nearest(model, options)
        model.focus = option
        #for option in options
            neighbors = get_neighbors(game, option)
            candidates = filter(x->!x.revealed, neighbors)
            if length(candidates) == i && any(x->!x.flagged, candidates)
                filter!(x->!x.flagged, candidates)
                choice = rand(candidates)
                flag_cell!(choice)
                model.focus = choice.idx
                game.mines_flagged += 1
                return nothing
            end
        #end
    end
    return nothing
end

function decide(model, game)
    flagged = findall(x->x.flagged, game.cells)
    if !isempty(flagged)
        option = select_nearest(model, flagged)
        model.focus = option
        selection = inspect(game, model, option)
        if !isempty(selection)
            model.focus = selection[1]
            return selection
         end
    end
    return typeof(flagged)()
end

function inspect(game, model, flagged)
    selection = CartesianIndex{2}[]
    for i in 1:model.depth
        neighbors = get_neighbors(game, flagged)
        filter!(x->x.revealed && (x.mine_count==i), neighbors)
        StatsBase.shuffle!(neighbors)
        for neighbor in neighbors
            fneighbors = get_neighbors(game, neighbor.idx)
            n_flagged = count(x->x.flagged, fneighbors)
            n_flagged ≠ i ? (continue) : nothing
            filter!(x->!x.revealed && !x.flagged, fneighbors)
            if !isempty(fneighbors)
                return [rand(fneighbors).idx]
            end
        end
    end
    return selection
end

function search(game::Game, n_mines)
    return findall(x->x.revealed && (x.mine_count == n_mines), game.cells)
end

function select_random_cell(game)
    unselected = findall(x->!x.revealed && !x.flagged, game.cells)
    return rand(unselected)
end

distance(c1::Cell, c2::Cell) = distance(c1.idx, c2.idx)
distance(c1, c2) = norm(Tuple(c1 - c2))

focus_probs(d, model) = exp.(-model.ϕ*d)/sum(exp.(-model.ϕ*d))

function select_nearest(model, coords)
    dist = map(x->distance(model.focus, x), coords)
    p = focus_probs(dist, model)
    return sample(coords, Weights(p))
end

function run!(game, model; realtime=false)
    choice = select_random_cell(game)
    model.focus = choice
    detonated = select!(game, choice)
    realtime ? (sleep(1);println(game)) : nothing
    while !detonated
        if rand() <= model.θguess
            choice = select_random_cell(game)
            model.focus = choice
            detonated = select!(game, choice)
            println("random selection")
        elseif rand() <= model.θflag
            flag_mines!(model, game)
        else
            choice = decide(model, game)
            isempty(choice) ? nothing : select!(game, choice[1])
        end
        realtime ? (sleep(1);println(game)) : nothing
        game_over(game) ? nothing : (break)
        game.trials += 1
        println(model.focus)
    end
    compute_score!(game)
    return nothing
end
