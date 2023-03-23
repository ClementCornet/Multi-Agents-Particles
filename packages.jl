using Pkg

dependencies = [
    "YAML",
    "Agents",
    "CellListMap",
    "StaticArrays",
    "InteractiveDynamics",
    "CairoMakie",
    "Statistics",
    "LinearAlgebra",
    "GLMakie",
    "CSV",
    "DataFrames"
]

Pkg.add(dependencies)