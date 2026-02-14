# System prompt for the medical AI assistant
MEDICAL_SYSTEM_PROMPT = """You are a helpful Medical AI assistant. Answer medical questions based ONLY on the provided context from medical textbooks.

Rules:
- Use only the information from the context provided
- If the answer is not in the context, say "I don't have enough information in my medical database to answer this question."
- Keep answers concise (3-5 sentences maximum)
- Use simple, clear language
- Never make up medical information"""


# User prompt template
def get_user_prompt(context, question):
    """Generate user prompt with context and question"""
    return f"""Context from medical textbooks:
{context}

Question: {question}

Please provide a concise answer based on the context above."""


# Function to create messages for HuggingFace chat models
def create_medical_chat_messages(context, question):
    """
    Create chat messages for HuggingFace InferenceClient

    Args:
        context: Retrieved context from medical documents
        question: User's medical question

    Returns:
        List of message dictionaries
    """
    return [
        {
            "role": "system",
            "content": MEDICAL_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": get_user_prompt(context, question)
        }
    ]
