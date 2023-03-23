using YAML

config = YAML.load_file("config.yaml")

CIRCLE = config["brownian_motion"]["circle"]
CROSS = config["brownian_motion"]["cross"]
WALLS = config["brownian_motion"]["walls"]
SQUARE = config["brownian_motion"]["square"]

using Agents
using CellListMap.PeriodicSystems
using StaticArrays

using InteractiveDynamics
using CairoMakie
using Statistics: mean, std, var
using LinearAlgebra
using WGLMakie

