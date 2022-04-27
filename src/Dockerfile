FROM arm64v8/ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

ARG FUNCTION_DIR="/home/app/"

# Required ArmNN software
ARG ARMNN_VERSION="v22.02"
ARG ARMNN_BINARY_URL="https://github.com/ARM-software/armnn/releases/download"
ARG PYARMNN_GIT_URL="https://github.com/ARM-software/armnn.git"

# Example image classification model
ARG ONNX_MODEL_URL="https://raw.githubusercontent.com/aws-samples/aws-graviton-ml-inference-armnn-example/main/data"

RUN set -e \
    ### Install prerequisite packages
    && apt update \
    && apt install -y --no-install-recommends git wget unzip python3 python3-dev python3-pip swig build-essential cmake autoconf autotools-dev automake libtool \
    ### Install Lambda runtime
    && mkdir -p ${FUNCTION_DIR} \
    && pip3 install awslambdaric --target ${FUNCTION_DIR} \
    ### Install required modules for ML inference code
    && pip3 install requests numpy Pillow \
    ### Download and extract ArmNN library binary
    && cd /tmp \
    && wget ${ARMNN_BINARY_URL}/${ARMNN_VERSION}/ArmNN-linux-aarch64.tar.gz \
    && tar -xf ArmNN-linux-aarch64.tar.gz -C /usr/lib/aarch64-linux-gnu \
    ### Download PyArmNN source code
    && git clone --depth=1 -b ${ARMNN_VERSION} ${PYARMNN_GIT_URL} \
    ### Build PyArmNN source code
    && cd /tmp/armnn/python/pyarmnn \
    && export ARMNN_INCLUDE=/tmp/armnn/include:/tmp/armnn/profiling/common/include \
    && python3 swig_generate.py -v \
    && python3 setup.py build_ext --inplace \
    && python3 setup.py bdist_wheel \
    && pip3 install dist/pyarmnn-28.0.0-cp38-cp38-linux_aarch64.whl \
    ### Download example ML model and labels
    && mkdir /onnx_model \
    && cd /onnx_model \
    && wget ${ONNX_MODEL_URL}/resnet50-v1-7.onnx \
    && wget ${ONNX_MODEL_URL}/synset.txt \
    ### Clean-up
    && cd /tmp \
    && rm -R armnn \
    && rm ArmNN-linux-aarch64.tar.gz \
    && apt auto-remove -y git wget unzip python3-dev swig build-essential cmake autoconf autotools-dev automake libtool \
    && apt clean all

# Add custom ML inference code
COPY app.py ${FUNCTION_DIR}

WORKDIR ${FUNCTION_DIR}
ENTRYPOINT [ "/usr/bin/python3", "-m", "awslambdaric" ]
CMD [ "app.handler" ]
