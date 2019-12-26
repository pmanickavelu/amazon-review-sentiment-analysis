# Train Simple LSTM Model
```
cd simple_lstm
python3.7 train.py
```
## This will generate the follwing files
* model.h5
* model.json
* tokenizer.pkl


# Model Validation
```
python3.7 validat.py
```

# Run the Model on you infrastructure
```
python3.7 web_service.py
```


# Run the Pre built Model on Docker 
```
docker run -it -p 5000:5000 docker.pkg.github.com/pmanickavelu/twitter-sentiment-analysis/simple_lstm
```

# Build your own Docker image
```
docker build -t <username>\<reponame> .
```


# Deploy on kubernetes
```
kubectl apply -f ReplicationController.yaml -f Service.yml
```
# Add AutoScaling to the service
```
kubectl apply -f HorizontalPodAutoscaler.yml
```