FROM python:3.11-slim

COPY ./requirements_production.txt .

RUN pip install -r requirements_production.txt

COPY app ./app
COPY BurrahMobileNet.onnx .

EXPOSE 5911

CMD [ "python" , "app/routes.py"]
#CMD ["uv", "run", "src/app/routes.py"]



