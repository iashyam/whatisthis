# Vision Models

I have veen experimenting with Vision Models for quite a while now. I am trying to implement them from scartch to learn more about the neural netowrk architecture. I am also doing this to create a full end to end project, complete with automatic testing and deployment. I have done these models now

- MobileNet V2: Paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381)

TODO:

- [x] Containerize the model with Docker. 
- [ ] Put automated testing with github actions. 
- [ ] Deploy the model to server. 

How to use it from Docker

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
