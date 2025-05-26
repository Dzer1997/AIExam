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
        return "You're a senior data scientist with a sharp eye for valuable real estate insights. " \
        "Outline the 15 most impactful engineered features to improve a house price prediction model, " \
        "using both traditional inputs like square footage and creative external sources such as school " \
        "ratings or walkability scores. For each feature, " \
        "explain why it helps predict price more accurately—no fluff, just actionable ideas."
    elif mode == "Detailed":
        return "Design the ideal machine learning pipeline for building a high-performance house price prediction model that can scale on cloud infrastructure. " \
        "Walk through each stage, from raw data ingestion to preprocessing, model selection, validation, and deployment. " \
        "You’re confident, bold, and a little unimpressed with over-engineered fluff—focus on what actually works and why."
    elif mode == "Market-Based (RAG)":
        return "You specialize in writing killer prompts for LLMs used in real estate data science. Come up with 5 high-impact prompts that would help " \
        "an LLM generate code or explanations for building tree-based models (like XGBoost or Random Forest) " \
        "to predict housing prices. Be creative, specific, and do not be afraid to push the boundaries of what prompt engineering can do."
    elif mode == "AI Analyst":
        return "You're reviewing a house price prediction model that underperforms despite good training data. " \
        "After seeing the outputs, you are not impressed. Give three sharp, clear reasons why the model might be struggling," \
        " and recommend practical next steps to debug or improve it. No sugarcoating—just straight, " \
        "helpful advice with examples where possible."
    elif mode == "Buyer-Friendly":
        return "Take a $750,000 house price prediction and break it down using SHAP or LIME in a way a homeowner could understand. " \
        "Show how each feature (like location, size, and condition) " \
        "contributed to the final prediction, using simple, non-technical language. The goal is clarity and trust—explain " \
        "the math without sounding like a machine."
    elif mode == "Investor Focus":
        return "Write a clean, modular Python script using XGBoost to predict house prices based on a structured dataset, " \
        "like one you did find on Kaggle. Include comments that explain each function s role in the pipeline. You are a senior developer who s " \
        "done this a hundred times, and you hate messy code, so keep it elegant, well-documented, and ready for production."
    elif mode == "Custom":
         return rag_context
    else:
        return "Explain the predicted price based on the data."