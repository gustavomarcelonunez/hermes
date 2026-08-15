import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Available models (internal keys)
# -----------------------------
MODEL_CONFIG = {
    "openai_gpt4omini": {
        "provider": "openai",
        "model": "gpt-4o-mini"
    },
    "ministral-8b-2512": {
            "provider": "mistral",
            "model": "ministral-8b-2512"
    },
    "ministral-14b-2512": {
        "provider": "mistral",
        "model": "ministral-14b-2512"
    },
    "mistral-small-2506": {
        "provider": "mistral",
        "model": "mistral-small-2506"
    },
    "mistral-large-2512": {
        "provider": "mistral",
        "model": "mistral-large-2512"
    },
    "openai/gpt-oss-120b": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b"
    },
    "openai/gpt-oss-20b": {
        "provider": "groq",
        "model": "openai/gpt-oss-20b"
    },
    "groq/compound": {
        "provider": "groq",
        "model": "groq/compound"
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "model": "gemini-2.5-flash"
    },

}

# -----------------------------
# API Keys
# -----------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------------
# Utility: Run chat for any provider
# -----------------------------
def run_llm(model_key, system_prompt, user_prompt):
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model key: {model_key}")

    cfg = MODEL_CONFIG[model_key]
    provider = cfg["provider"]
    model = cfg["model"]

    if provider == "openai":
        return _run_openai(model, system_prompt, user_prompt)

    elif provider == "mistral":
        return _run_mistral(model, system_prompt, user_prompt)
    
    elif provider == "groq":
        return _run_groq(model, system_prompt, user_prompt)

    elif provider == "gemini":
        return _run_gemini(model, system_prompt, user_prompt)

    else:
        raise ValueError(f"Provider {provider} not implemented.")


# -----------------------------
# OPENAI backend
# -----------------------------
def _run_openai(model, system_prompt, user_prompt):
    client = OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


# -----------------------------
# MISTRAL backend
# -----------------------------
def _run_mistral(model, system_prompt, user_prompt):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    r = requests.post(url, json=payload, headers=headers)

    if r.status_code != 200:
        return f"[MISTRAL ERROR {r.status_code}] {r.text}"

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except:
        return f"[MISTRAL JSON ERROR] Raw: {r.text}"
    
# -----------------------------
# GROQ backend
# -----------------------------
def _run_groq(model, system_prompt, user_prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    r = requests.post(url, json=payload, headers=headers)

    if r.status_code != 200:
        return f"[GROQ ERROR {r.status_code}] {r.text}"

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except:
        return f"[GROQ JSON ERROR] Raw: {r.text}"


# -----------------------------
# GEMINI backend (v1beta OpenAI-compatible)
# -----------------------------
def _run_gemini(model, system_prompt, user_prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {GEMINI_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    r = requests.post(url, json=payload, headers=headers)

    if r.status_code != 200:
        return f"[GEMINI ERROR {r.status_code}] {r.text}"

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except:
        return f"[GEMINI JSON ERROR] Raw: {r.text}"