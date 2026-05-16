FROM dolfinx/dolfinx:v0.8.0

WORKDIR /workspace/simulation

# Extra packages on top of the DOLFINx base.
# The base already ships: numpy, scipy, matplotlib, mpi4py, petsc4py,
# dolfinx, ufl, basix, ffcx — nothing else is needed.
COPY requirements_simulation.txt .
RUN pip install --no-cache-dir -r requirements_simulation.txt

# Python source files are bind-mounted at runtime from the host.
# See docker-compose.yml:  ./simulation:/workspace/simulation
# No COPY needed — edits on host are immediately reflected in container.

# Pre-create data directories
RUN mkdir -p /workspace/data/snapshots /workspace/data/figures

CMD ["python3", "run_parametric.py"]
