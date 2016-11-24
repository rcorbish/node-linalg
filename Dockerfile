FROM ubuntu:16.04

RUN apt update && \
        apt -y install curl && \
        apt -y install xz-utils && \
        curl -L -s https://nodejs.org/dist/v7.2.0/node-v7.2.0-linux-x64.tar.xz -o node-v7.2.0-linux-x64.tar.xz  && \
	tar -xJf node-v7.2.0-linux-x64.tar.xz  && \
	rm node-v7.2.0-linux-x64.tar.xz  && \
	chown -R root:root /node-v7.2.0-linux-x64/bin/node && \
	ln -s /node-v7.2.0-linux-x64/bin/node /usr/bin && \
	ln -s /node-v7.2.0-linux-x64/bin/npm /usr/bin && \
        apt -y install python && \
        apt -y install libopenblas-dev && \
        apt -y install liblapacke-dev && \
        apt -y install make && \
        apt -y install g++ && \
	npm install lalg

VOLUME ["/github"]

CMD [ "/usr/bin/node" ]
#
# Run this using this command line
#
# docker run -v /home/docker/github:/github rcorbish/build-javacpp
#


