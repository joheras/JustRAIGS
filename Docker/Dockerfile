FROM --platform=linux/amd64  pytorch/pytorch

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN adduser --system --group user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app

RUN python -m pip install \
    --no-color \
    --requirement requirements.txt


#ADD --chown=user:user https://www.dropbox.com/scl/fi/srobfm3uuq4gmvf7824gh/efficientnetv2_rw_s.pkl?rlkey=m9wns34nam2ov0nqexdxl4q4h&dl=1 /opt/app/efficientnetv2_rw_s.pkl
ADD --chown=user:user https://www.dropbox.com/scl/fi/j7mbou4chzklqkqhq4q44/swin_base_patch4_window12_384.ms_in1k.pkl?rlkey=1kd8rpi8wdacmnkftv6ysavel&dl=1 /opt/app/swin_base_patch4_window12_384.ms_in1k.pkl
ADD --chown=user:user https://www.dropbox.com/scl/fi/3afueysuterseha4o4pi5/convnext_base_384_in22ft1k_multi_labelv3.pkl?rlkey=08unhmat8944ddfa9f0jst6a1&dl=1 /opt/app/convnext_base_384_in22ft1k_multi_labelv3.pkl
ADD --chown=user:user https://www.dropbox.com/scl/fi/6zt4j8jlaq791qxd63zx6/convnext_base_384_in22ft1k.pkl?rlkey=2mldxak4t67r72ngj8jkqw310&dl=1 /opt/app/convnext_base_384_in22ft1k.pkl


COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app

# RUN python /opt/app/download-model.py



# ENTRYPOINT ["ls","-l", "/opt/app/"]
ENTRYPOINT ["python", "inference.py"]
