# Medical AI Chatbot with RAG Architecture

A production-ready medical information chatbot that leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware health information. Built with open-source LLMs and deployed on AWS with a fully automated CI/CD pipeline.

## Overview

This application combines semantic search with large language models to deliver reliable medical information. By retrieving relevant medical literature before generating responses, it significantly reduces hallucination risks and ensures answers are grounded in factual content.

## Key Features

- **Retrieval-Augmented Generation**: Queries vector database before generating responses
- **Multi-Model Fallback**: Automatically tries multiple open-source LLMs for reliability
- **Semantic Search**: Uses Pinecone vector database for intelligent document retrieval
- **Production Deployment**: Containerized with Docker and deployed on AWS EC2
- **Automated CI/CD**: GitHub Actions pipeline for continuous deployment
- **Scalable Architecture**: Designed for production workloads

## Tech Stack

### Backend & ML

- **Flask**: Web framework for API endpoints
- **LangChain**: RAG orchestration and chain management
- **HuggingFace**: Open-source LLM inference (Mistral-7B, Llama-3.2, Phi-3)
- **Pinecone**: Vector database for semantic search
- **Sentence Transformers**: Text embeddings generation

### DevOps & Infrastructure

- **Docker**: Application containerization
- **AWS ECR**: Container registry
- **AWS EC2**: Production hosting
- **GitHub Actions**: CI/CD automation

## Architecture

```
User Query → Flask API → Pinecone Vector Search → LangChain RAG
                                ↓
                         Retrieved Documents
                                ↓
                    HuggingFace LLM (Multi-model fallback)
                                ↓
                         Generated Response
```

### RAG Pipeline

1. **Query Processing**: User question is converted to embeddings
2. **Vector Search**: Top-k relevant documents retrieved from Pinecone
3. **Context Building**: Retrieved documents combined with user query
4. **LLM Generation**: Open-source model generates contextual response
5. **Response Delivery**: Answer returned to user via Flask API

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- AWS Account (for cloud deployment)
- Pinecone Account
- HuggingFace Account (for API access)

## Installation

### Local Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd Medical_Chatbot_pinecone_Flask_AWS
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACE_API_KEY=your_huggingface_token
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=your_aws_region
```

5. **Run the application**

```bash
python app.py
```

The application will be available at `http://localhost:8080`

## Docker Deployment

### Build Docker Image

```bash
docker build -t medical-chatbot .
```

### Run Container Locally

```bash
docker run -d -p 8080:8080 \
  -e PINECONE_API_KEY=your_key \
  -e HUGGINGFACE_API_KEY=your_key \
  --name medical-chatbot \
  medical-chatbot
```

### Check Container Status

```bash
docker ps
docker logs medical-chatbot
```

## AWS Deployment

### Prerequisites

1. **AWS EC2 Instance**: Ubuntu 20.04+ with Docker installed
2. **AWS ECR Repository**: Created in your AWS account
3. **Security Group**: Inbound rule for port 8080
4. **GitHub Secrets**: Configure the following secrets in your repository:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_DEFAULT_REGION`
   - `ECR_REPO`
   - `EC2_HOST`
   - `EC2_USER`
   - `EC2_SSH_KEY`
   - `PINECONE_API_KEY`
   - `HUGGINGFACE_API_KEY`

### CI/CD Pipeline

The GitHub Actions workflow automatically:

1. **Build Phase**:
   - Checks out code
   - Authenticates with AWS ECR
   - Builds Docker image with layer caching
   - Pushes to ECR with retry logic

2. **Deploy Phase**:
   - Connects to EC2 via SSH
   - Installs AWS CLI if needed
   - Cleans up old Docker resources
   - Pulls latest image from ECR
   - Stops old container and starts new one

### Manual Deployment

```bash
# Build and push to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker build -t medical-chatbot .
docker tag medical-chatbot:latest <account-id>.dkr.ecr.<region>.amazonaws.com/medical-chatbot:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/medical-chatbot:latest

# SSH to EC2 and deploy
ssh -i your-key.pem ubuntu@your-ec2-ip
docker pull <account-id>.dkr.ecr.<region>.amazonaws.com/medical-chatbot:latest
docker run -d -p 8080:8080 --name my-app \
  -e PINECONE_API_KEY=$PINECONE_KEY \
  -e HUGGINGFACE_API_KEY=$HF_KEY \
  <account-id>.dkr.ecr.<region>.amazonaws.com/medical-chatbot:latest
```

## Project Structure

```
Medical_Chatbot_pinecone_Flask_AWS/
├── .github/
│   └── workflows/
│       └── cicd.yaml           # CI/CD pipeline configuration
├── src/
│   ├── helper.py               # Embeddings and utility functions
│   └── prompt.py               # LLM prompt templates
├── templates/
│   └── chat.html               # Frontend chat interface
├── static/                     # Static assets (CSS, JS)
├── app.py                      # Main Flask application
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .dockerignore              # Docker ignore rules
└── README.md                   # This file
```

## API Endpoints

### GET `/`

Returns the chat interface HTML page.

### POST `/get`

Handles chat requests.

**Request Body:**

```json
{
  "msg": "What are the symptoms of diabetes?"
}
```

**Response:**

```json
{
  "answer": "Based on the medical literature, common symptoms of diabetes include..."
}
```

## Usage Example

```python
import requests

url = "http://your-ec2-ip:8080/get"
data = {"msg": "What are the symptoms of diabetes?"}

response = requests.post(url, json=data)
print(response.json()["answer"])
```

## Configuration

### LLM Models

The application attempts models in this order:

1. Mistral-7B-Instruct-v0.2
2. Mistral-7B-Instruct-v0.3
3. Phi-3-mini-4k-instruct
4. Zephyr-7B-beta
5. Llama-3.2-3B-Instruct

Modify the `models_to_try` list in `app.py` to customize.

### Vector Search Parameters

Adjust retrieval settings in `app.py`:

```python
retriever = docsearch.as_retriever(search_kwargs={"k": 3})  # Number of documents
```

### LLM Parameters

Customize generation in `ask_medical_question()`:

```python
response = hf_client.chat_completion(
    messages=messages,
    model=model,
    max_tokens=512,      # Maximum response length
    temperature=0.3      # Creativity (0.0-1.0)
)
```

## Troubleshooting

### Container Not Running

```bash
docker ps -a                    # Check container status
docker logs my-app              # View container logs
docker restart my-app           # Restart container
```

### Out of Disk Space

```bash
docker system prune -af --volumes  # Clean up Docker resources
df -h                              # Check disk usage
```

### Connection Refused

- Verify security group allows inbound traffic on port 8080
- Check if container is running: `docker ps`
- View logs: `docker logs my-app`

### API Key Issues

- Verify environment variables are set correctly
- Check HuggingFace token has inference API access
- Ensure Pinecone API key is valid

## Performance Optimization

### Caching

Consider implementing caching for:

- Embedding generation
- Vector search results
- LLM responses for common queries

### Production Deployment

For production workloads, consider:

- Using Gunicorn or uWSGI instead of Flask dev server
- Implementing request rate limiting
- Adding monitoring and logging (CloudWatch, Datadog)
- Setting up auto-scaling with AWS ECS/EKS
- Using a CDN for static assets

## Security Considerations

- Never commit `.env` files or secrets to version control
- Use AWS Secrets Manager for production credentials
- Implement API authentication and rate limiting
- Keep dependencies updated for security patches
- Use HTTPS in production environments

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for RAG framework
- HuggingFace for open-source models
- Pinecone for vector database
- Medical literature providers for knowledge base

## Author

**Muhammad Azan Afzal**

For questions or support, please open an issue in the repository.

---

**Built with modern ML engineering practices and deployed on AWS infrastructure.**
