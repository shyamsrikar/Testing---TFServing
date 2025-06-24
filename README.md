# Testing - Tensor Flow Serving

## What is TFServing?

- TensorFlow Serving is a tool that helps you deploy and serve machine learning models in real-time using REST or gRPC APIs. 
- It allows you to easily load, update, and get predictions from trained models in production environments.

![Screenshot from 2025-06-22 17-20-09](https://github.com/user-attachments/assets/8aabf8a6-65d6-479b-abef-2d84025fb90f)



## Before installing TFServing, you need a server to run it. Here’s how to set up an Ubuntu server on AWS:

- 1) Login to AWS Console and go to EC2 Dashboard.
     
- 2) Click "Launch Instance".
     
![Screenshot from 2025-06-24 09-28-26](https://github.com/user-attachments/assets/4ee53e46-979b-44a2-868e-adf0bd7fa825)


- 3) Choose AMI: Select Ubuntu Server 24.04 LTS (HVM), SSD Volume Type.

![Screenshot from 2025-06-24 09-28-42](https://github.com/user-attachments/assets/f025aca3-b6c2-4f32-9b94-ec10af582dc2)

- 4) Instance Type: Choose t2.micro (free tier) or a higher type if needed.

     
- 5) Key Pair: Select or create a key pair to access your server.

![Screenshot from 2025-06-24 09-29-05](https://github.com/user-attachments/assets/3236297e-d30a-4c34-b175-2c669b39591a)

- 6) Network Settings:
Allow SSH (port 22)
Allow Custom TCP on port 8501 (needed for TFServing)     

![Screenshot from 2025-06-24 09-30-21](https://github.com/user-attachments/assets/127779d6-e12c-464e-9c38-1f5cdc37f088)

- 7) Launch Instance.
     
- 8) After it’s running, connect to it using:

![Screenshot from 2025-06-24 09-32-13](https://github.com/user-attachments/assets/1a2313fd-8758-4bf0-974d-f6fac5396d66)

## Replace with actual public ip of instance
```
 ssh -i your-key.pem ubuntu@<your-public-ip>
```

After Connecting:

![Screenshot from 2025-06-24 09-31-33](https://github.com/user-attachments/assets/0fd9405d-6fd7-472c-8297-c469ac5b0f3f)


# 🛠️ Installation Steps for TFServing on Ubuntu Server After Connecting to the Instance

## To install TFServing on Ubuntu, follow these steps:


### ✅ Step 1: Update System

Run a command to update the list of software packages and upgrade them to the latest version.

```
 sudo apt update && sudo apt upgrade -y
```

### ✅ Step 2: Install Docker & Docker Compose

Run the below command to install docker and docker compose and in a python env

```
sudo apt install docker.io docker-compose python3-venv -y

```

### ✅ Step 3: Enable Docker service:

```
sudo systemctl enable docker
sudo systemctl start docker

```

![Screenshot from 2025-06-24 09-34-30](https://github.com/user-attachments/assets/382a6c8a-8fb0-4a97-afde-5845988562ce)


### ✅ Step 4: Create Project Directory and Virtual Environment


```
mkdir -p ~/tfserving/models/half_plus_two/1
cd ~/tfserving
python3 -m venv venv
source venv/bin/activate

```

### ✅ Step 5: Install TensorFlow and Saving the Model 

- Install TensorFlow in the virtual environment:

```
pip install tensorflow

```

![Screenshot from 2025-06-24 09-37-20](https://github.com/user-attachments/assets/75b6aec7-cd66-4809-82de-864c9f0c4628)


- Create and save the model using Python:

```
python

```
- Then paste this Python code:  

```
import tensorflow as tf

class HalfPlusTwo(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, x):
        return 0.5 * x + 2

model = HalfPlusTwo()
tf.saved_model.save(model, "models/half_plus_two/1")
exit()

```

### ✅ Step 6: You should now see :

- saved_model.pb variables/
  
```
ls models/half_plus_two/1

```

### ✅ Step 7: Deactivate the venv:

```
deactivate

```

### ✅ Step 8: Create docker-compose.yml for installing TFServing

Still inside ~/tfserving directory:

```
nano docker-compose.yml

```

### ✅ Step 9: Paste this content below:

```
version: '3.7'

services:
  tf_serving:
    image: tensorflow/serving
    ports:
      - 8501:8501
    volumes:
      - ./models:/models
    environment:
      - MODEL_NAME=half_plus_two

```
- Save and exit (Ctrl+O → Enter → Ctrl+X)

### ✅ Step 10:  Start TensorFlow Serving

```
sudo docker-compose up -d

```

![Screenshot from 2025-06-24 09-39-08](https://github.com/user-attachments/assets/8ee00446-536a-46de-aa43-98569615ed66)


- Verify it’s running:

```
sudo docker ps
```
You should see a container with port 8501 exposed.

### ✅ Step 11: Access via Public IP

- Find your public IP:

```
curl ifconfig.me

```

- Test model availability:

```
curl http://your-public-ip:8501/v1/models/half_plus_two

```

![Screenshot from 2025-06-24 09-40-12](https://github.com/user-attachments/assets/8406858f-00cf-4fba-8ada-5726f6271eb0)


![Screenshot from 2025-06-24 09-40-39](https://github.com/user-attachments/assets/8fd4203b-960c-412c-920b-c8898c05559f)

- Test inference:

```
curl -X POST http://your-public-ip:8501/v1/models/half_plus_two:predict \
  -H "Content-Type: application/json" \
  -d '{"signature_name":"serving_default", "instances":[1.0, 2.0, 5.0]}'
```

- ✅ Expected output:

```
{"predictions":[2.5, 3.0, 4.5]}

```

![Screenshot from 2025-06-24 09-41-27](https://github.com/user-attachments/assets/a6e39a82-7c0b-45c2-be69-8727111bdfc5)



# Monitors Implemented for Testing TFServing service:

# Monitors Implemented:  

![Screenshot from 2025-06-24 09-45-02](https://github.com/user-attachments/assets/7d376781-672b-4eb5-8887-814e304b8a51)


### Model Availability Check:

- Continuously verifies the availability of the deployed TensorFlow model to ensure it is in a serving-ready state and able to handle inference requests without failure.

![Screenshot from 2025-06-24 09-45-43](https://github.com/user-attachments/assets/1f59f179-6bd3-4553-ab25-577b3048c9db)

  
### Metadata Integrity Check:

- Validates the integrity of the model metadata such as signature definitions and version consistency to ensure that the correct model is being served.

![Screenshot from 2025-06-24 09-45-58](https://github.com/user-attachments/assets/01faf046-f9be-4ca3-b764-fe2a7f57f6a9)


### Inference Latency Check:

- Measures the response time of inference requests to detect performance bottlenecks or degraded latency in model serving.

![Screenshot from 2025-06-24 09-46-15](https://github.com/user-attachments/assets/59212b61-5511-45c4-8676-681e70b0611c)


### Connectivity Check:

- Verifies network-level connectivity to the TensorFlow Serving endpoint, ensuring that the service is reachable and accessible from the monitoring host.

![Screenshot from 2025-06-24 09-46-26](https://github.com/user-attachments/assets/e6db58b6-8fe4-4893-9700-c0ea963a872e)



# Conclusion:

- All the monitors are running successfully after passing the environment variables for TFServing service.






     
