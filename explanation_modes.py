def list_prompt_modes():
    return [
        "Simple",
        "Detailed",
        "Market-Based (RAG)",
        "AI Analyst",
        "Buyer-Friendly",
        "Investor Focus",
        "Custom"
    ]

def get_prompt_template(mode, rag_context=""):
    if mode == "Simple":
        return "You're a senior data scientist with a sharp eye for valuable real estate insights. Outline the 15 most impactful engineered features to improve a house price prediction model, using both traditional inputs like square footage and creative external sources such as school ratings or walkability scores. For each feature, explain why it helps predict price more accurately—no fluff, just actionable ideas."
    elif mode == "Detailed":
        return "Design the ideal machine learning pipeline for building a high-performance house price prediction model that can scale on cloud infrastructure. Walk through each stage, from raw data ingestion to preprocessing, model selection, validation, and deployment. You’re confident, bold, and a little unimpressed with over-engineered fluff—focus on what actually works and why."
    elif mode == "Market-Based (RAG)":
        return f"Use this market context and house data to explain the price:\n\n{rag_context}"
    elif mode == "AI Analyst":
        return "You are a real estate AI analyst. Use the provided information to justify the price."
    elif mode == "Buyer-Friendly":
        return "Explain this price in a friendly way to a first-time homebuyer."
    elif mode == "Investor Focus":
        return f"Explain the price from an investor's point of view.\n\nMarket Context:\n{rag_context}"
    elif mode == "Custom":
        return ""
    else:
        return "Explain the predicted price based on the data."