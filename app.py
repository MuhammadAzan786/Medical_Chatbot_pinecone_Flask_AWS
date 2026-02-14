from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from src.prompt import create_medical_chat_messages
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

if HUGGINGFACE_API_KEY:
    os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
else:
    raise ValueError(
        "HUGGINGFACE_API_KEY not found in .env file. Please add it!")

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone Vector Store
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# Initialize HuggingFace Inference Client
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)


def ask_medical_question(question, context_docs):
    """
    Ask a medical question to the AI agent with retrieved context

    Args:
        question: User's medical question
        context_docs: Retrieved documents from Pinecone

    Returns:
        AI-generated answer
    """

    # Combine context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in context_docs])

    # Create messages using prompts from prompt.py
    messages = create_medical_chat_messages(context, question)

    # Try models in order of preference (compatible with new infrastructure)
    models_to_try = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-4k-instruct",
        "HuggingFaceH4/zephyr-7b-beta",
        "meta-llama/Llama-3.2-3B-Instruct"
    ]

    for model in models_to_try:
        try:
            # Try chat_completion first (preferred method)
            response = hf_client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=512,
                temperature=0.3
            )
            print(f"✓ Using model: {model}")
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            if "not supported" in error_msg.lower() or "nscale" in error_msg.lower():
                # Try text_generation as fallback
                try:
                    prompt = f"""You are a helpful Medical AI assistant. Answer based ONLY on the context provided.

Context: {context}

Question: {question}

Answer (3-5 sentences):"""

                    response = hf_client.text_generation(
                        prompt,
                        model=model,
                        max_new_tokens=512,
                        temperature=0.3,
                        return_full_text=False
                    )
                    print(f"✓ Using model (text_generation): {model}")
                    return response
                except:
                    continue
            continue

    # If all models fail, provide helpful error
    raise Exception(
        "All models failed. Try one of these solutions:\n"
        "1. Check your HuggingFace API token is valid\n"
        "2. You may be rate limited (free tier has limits)\n"
        "3. Try again in a few minutes\n"
        "4. Consider using a Pro HuggingFace account for better access"
    )

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    """Handle chat requests from the user"""
    try:
        # Get the user's question from the request
        msg = request.json.get("msg")

        if not msg:
            return jsonify({"answer": "Please provide a question."})

        print(f"User Question: {msg}")

        # Retrieve relevant documents from Pinecone
        context_docs = retriever.get_relevant_documents(msg)
        print(f"Retrieved {len(context_docs)} relevant documents")

        # Get answer using HuggingFace model
        answer = ask_medical_question(msg, context_docs)

        print(f"Answer: {answer}")

        return jsonify({"answer": answer})

    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        return jsonify({"answer": error_message})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
