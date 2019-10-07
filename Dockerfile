FROM python:3.7

#RUN apk --no-cache add --update python-dev gfortran build-base freetype-dev libpng-dev openblas-dev jpeg-dev zlib-dev openjpeg-dev tiff-dev

COPY requirements.txt /code/requirements.txt

WORKDIR /code
RUN pip install -r requirements.txt


COPY . /code

CMD ["python"]