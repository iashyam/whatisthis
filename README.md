# Whatisthis

Implemention and training of different Vision models are from scratch and use it to label any image.  

## MobileNet

- MobileNet V2: Paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381)

TODO:

- [x] Containerize the model with Docker. 
- [x] Put automated testing with github actions. 
- [ ] Deploy the model to server. 
- [x] Write a ETL pipeline. 
- [x] Write training loop. 
- [ ] Write documentation for ETL.

### Using through Docker

First you need to pull image from dockerhub

```bash
docker pull dokckerhub.com/iashyam/whatisthis:latest
```

Then just run, remember to map the port `5000`:

```bash
docker run -p 5000:5000 dockerhub.com/iashyam/whatisthis:latest
```

Enjoy the demo at `localhost:5000`.

## Contributions

This is a personal project and I am not recieving any contributions as per now. 

I am on to efficient net next. 
