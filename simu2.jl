

using Agents
using LinearAlgebra
using YAML

config = YAML.load_file("config.yaml")


# Zone config
CIRCLE = config["zone"]["circle"] #y
CROSS = config["zone"]["cross"] #y
SQUARE = config["zone"]["square"] #y
WIGGLE_ANGLE = config["zone"]["wiggle_angle"] #n

# Model config
MODEL_SIDES = config["model"]["side_length"] #y
WALLS = config["model"]["walls"] #n

# Particle config
N = config["particles"]["number"] #y
COLLISIONS = config["particles"]["collisions"] #y
MAX_RADIUS = config["particles"]["max_radius"] #y
FIXED_RADIUS = config["particles"]["fixed_radius"] #n
FIXED_MASS = config["particles"]["fixed_mass"] #n
FIXED_K = config["particles"]["fixed_k"] #n

# Particle initialization
START_ZERO_X = config["particles"]["start_zero_x"] #n

# Render simulation
VIDEO = config["simulation"]["output_video"] #y
INTERACTIVE_WINDOW = config["simulation"]["interactive_window"] #y


@agent Particle ContinuousAgent{2} begin
    r::Float64 # radius
    k::Float64 # repulsion force constant
    mass::Float64
end
Particle(; id, pos, vel, r, k, mass) = Particle(id, pos, vel, r, k, mass)


using CellListMap.PeriodicSystems
using StaticArrays

function initialize_model(;
    number_of_particles=N,
    sides=SVector(MODEL_SIDES,MODEL_SIDES),
    dt=0.001,
    max_radius=MAX_RADIUS,
    parallel=true
)
    ## initial random positions
    positions = [sides .* rand(SVector{2,Float64}) for _ in 1:number_of_particles]

    ## We will use CellListMap to compute forces, with similar structure as the positions
    forces = similar(positions)

    ## Space and agents
    space2d = ContinuousSpace(Tuple(sides); periodic=true)

    ## Initialize CellListMap periodic system
    system = PeriodicSystem(
        positions=positions,
        unitcell=sides,
        cutoff=2 * max_radius,
        output=forces,
        output_name=:forces, # allows the system.forces alias for clarity
        parallel=parallel,
    )

    ## define the model properties
    ## The clmap_system field contains the data required for CellListMap.jl
    properties = (
        dt=dt,
        number_of_particles=number_of_particles,
        system=system,
    )
    model = ABM(Particle,
        space2d,
        properties=properties
    )

    ## Create active agents
    for id in 1:number_of_particles
        add_agent_pos!(
            Particle(
                id=id,
                r=((0.5 + 0.9 * rand()) * max_radius)*(1-Int(FIXED_RADIUS)) + MAX_RADIUS * Int(FIXED_RADIUS),
                k=(10 + 20 * rand()), # random force constants
                mass=10.0 + 100 * rand(), # random masses
                pos=Tuple(positions[id]),
                vel=(100 * randn(), 100 * randn()), # initial velocities
            ),
            model)
    end

    return model
end


function calc_forces!(x, y, i, j, d2, forces, model)
    pᵢ = model[i]
    pⱼ = model[j]
    d = sqrt(d2)
    if d ≤ (pᵢ.r + pⱼ.r) && COLLISIONS
        dr = y - x
        fij = 2 * (pᵢ.k * pⱼ.k) * (d2 - (pᵢ.r + pⱼ.r)^2) * (dr / d)
        forces[i] += fij
        forces[j] -= fij
    end
    return forces
end


function model_step!(model::ABM)
    ## Update the pairwise forces at this step
    map_pairwise!(
        (x, y, i, j, d2, forces) -> calc_forces!(x, y, i, j, d2, forces, model),
        model.system,
    )
    return nothing
end


function agent_step!(agent, model::ABM)
    id = agent.id
    dt = model.properties.dt
    ## Retrieve the forces on agent id
    f = model.system.forces[id]
    a = f / agent.mass
    ## Update positions and velocities
    v = SVector(agent.vel) + a * dt
    x = SVector(agent.pos) + v * dt + (a / 2) * dt^2
    x = normalize_position(Tuple(x), model)
    agent.vel = Tuple(v)


    #agent.vel = agent.vel ./ norm(agent.vel)


    if isin_circle(agent) || isin_cross(agent) || isin_square(agent)
        agent.vel = sincos(2π * rand(model.rng)) .* norm(agent.vel)
    end
    

    move_agent!(agent, x, model)
    ## !!! IMPORTANT: Update positions in the CellListMap.PeriodicSystem
    model.system.positions[id] = SVector(agent.pos)
    return nothing
end



function isin_circle(agent)
    dist_center = sqrt((agent.pos[1]-(MODEL_SIDES/2))^2 + (agent.pos[2]-(MODEL_SIDES/2))^2)
    if dist_center <= Float64(CIRCLE)
        return true
    end
    return false
end

function isin_cross(agent)

    dist_x = abs(agent.pos[1] -(MODEL_SIDES/2))
    if dist_x <= Float64(CROSS)
        return true
    end
    dist_y = abs(agent.pos[2] -(MODEL_SIDES/2))
    if dist_y <= Float64(CROSS)
        return true
    end
    return false
end

function isin_square(agent)
    diff_x = abs(agent.pos[1] -(MODEL_SIDES/2))
    diff_y = abs(agent.pos[2] -(MODEL_SIDES/2))
    if diff_x <= Float64(SQUARE) && diff_y <= Float64(SQUARE)
        return true
    end
    return false
end

function acolor(agent)
    if agent.mass == Inf
        return "black"
    end
    if isin_circle(agent)
        return "red"
    end
    if isin_cross(agent)
        return "red"
    end
    if isin_square(agent)
        return "red"
    end
    return "blue"
end


# Which should be quite fast
model = initialize_model()
using InteractiveDynamics
using CairoMakie
CairoMakie.activate!() # hide
model = initialize_model(number_of_particles=N)

if VIDEO
    println("... Generating simulation video ...")
    abmvideo(
        "simu2.mp4", model, agent_step!, model_step!;
        framerate=20, frames=200, spf=5,
        title="youpi",
        as=p -> p.r, # marker size
        ac=acolor # marker color
    )
    println("Done.")
elseif INTERACTIVE_WINDOW
    println("oui bah c'est bon je rajoute ça")
else
    run!(model, agent_step!,model_step!, 1000;)
end
