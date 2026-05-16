FROM python:3.11-slim

WORKDIR /workspace/data_driven

# build-essential  — compiles C extensions (pytensor, jaxlib wheels may need it)
# libgomp1         — OpenMP for scikit-learn / JAX CPU
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_data_driven.txt .
RUN pip install --no-cache-dir -r requirements_data_driven.txt

# Python source files are bind-mounted at runtime from the host.
# See docker-compose.yml:  ./data_driven:/workspace/data_driven
# No COPY needed — edits on host are immediately reflected in container.

RUN mkdir -p /workspace/data/rom \
             /workspace/data/posterior/figures

# Streamlit headless config
RUN mkdir -p /root/.streamlit && \
    printf '[general]\nemail = ""\n' \
        > /root/.streamlit/credentials.toml && \
    printf '[server]\nheadless = true\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n' \
        > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["bash", "-c", "python3 train_rom.py && python3 bayesian_inference.py && python3 field_reconstruction.py && python3 posterior_analysis.py && streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
