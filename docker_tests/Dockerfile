FROM ubuntu:20.04

# install debian packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        pip \
        python3 \
        wget

# install julia and numpy
RUN wget --quiet https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.4-linux-x86_64.tar.gz && \
    tar -zxf julia-1.6.4-linux-x86_64.tar.gz && \
    mv julia-1.6.4 /opt && \
    ln -s /opt/julia-1.6.4/bin/julia /bin/julia

CMD ["bash"]
ENTRYPOINT ["python3", "tests.py"]

## install the package
WORKDIR /CherenkovDeconvolution_jl
ADD .DEPS/ ./
RUN pip install -e .

# add the tests at the very end to speed up builds for which only the tests need to be updated
ADD tests.py ./
