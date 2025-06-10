FROM python:3.11
RUN apt update && \ 
    apt install -y --no-install-recommends graphviz libblas-dev liblapack-dev gfortran && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip
COPY .inputr[c] .bashr[c] requirements.txt /root/
RUN pip install --no-cache-dir -r /root/requirements.txt
ARG USERNAME
USER root
COPY .env /tmp/
RUN . /tmp/.env && rm /tmp/.env && \
    sed -i 's/^UID_MAX.*/UID_MAX 2147483647/' /etc/login.defs && \
    sed -i 's/^GID_MAX.*/GID_MAX 2147483647/' /etc/login.defs && \
    if getent passwd $USERNAME; then userdel -f $USERNAME; fi && \
    if getent group $GROUPNAME; then groupdel $GROUPNAME; fi && \
    if getent group $GROUP_GID; then TMP_NAME=$(getent group $GROUP_GID | cut -d: -f1); groupdel $TMP_NAME; fi && \
    groupadd --gid $GROUP_GID $GROUPNAME && \
    useradd --uid $USER_UID --gid $GROUP_GID -m $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL >> /etc/sudoers
USER $USERNAME
WORKDIR /workspaces/scenic-ins