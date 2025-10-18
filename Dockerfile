FROM limbo018/dreamplace:cuda
LABEL maintainer="Xi Lin <lnx@smail.nju.edu.cn>"

# Install python dependencies. 
RUN pip install \
        networkit>=11.0
