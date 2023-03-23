

using Agents

using YAML

config = YAML.load_file("config.yaml")


# Brownian Motion config
CIRCLE = config["brownian_motion"]["circle"]
CROSS = config["brownian_motion"]["cross"]
SQUARE = config["brownian_motion"]["square"]

# Model config
MODEL_SIDES = config["model"]["side_length"]
WALLS = config["model"]["walls"]

# Particle config
N = config["particles"]["number"]
COLLISIONS = config["particles"]["collisions"]
MAX_RADIUS = config["particles"]["max_radius"]
FIXED_RADIUS = config["particles"]["fixed_radius"]
FIXED_MASS = config["particles"]["fixed_mass"]
FIXED_K = config["particles"]["fixed_k"]

# Particle initialization
START_ZERO_X = config["particles"]["start_zero_x"]

# Render simulation
VIDEO = config["simulation"]["output_video"]
INTERACTIVE_WINDOW = config["simulation"]["interactive_window"]


@agent Particle ContinuousAgent{2} begin
    r::Float64 # radius
    k::Float64 # repulsion force constant
    mass::Float64
end
Particle(; id, pos, vel, r, k, mass) = Particle(id, pos, vel, r, k, mass)


using CellListMap.PeriodicSystems
using StaticArrays

function initialize_model(;
    number_of_particles=10_000,
    sides=SVector(500.0, 500.0),
    dt=0.001,
    max_radius=10.0,
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
                r=(0.5 + 0.9 * rand()) * max_radius,
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
    if d ≤ (pᵢ.r + pⱼ.r)
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
    move_agent!(agent, x, model)
    ## !!! IMPORTANT: Update positions in the CellListMap.PeriodicSystem
    model.system.positions[id] = SVector(agent.pos)
    return nothing
end

# Which should be quite fast
model = initialize_model()



using InteractiveDynamics
using CairoMakie
CairoMakie.activate!() # hide
model = initialize_model(number_of_particles=N)

if VIDEO
    abmvideo(
        "simu2.mp4", model, agent_step!, model_step!;
        framerate=20, frames=200, spf=5,
        title="Bouncing particles with CellListMap.jl acceleration",
        as=p -> p.r, # marker size
        ac=p -> p.k # marker color
    )
end
