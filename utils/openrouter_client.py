# utils/openrouter_client.py

import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from .prompt_manager import AVAILABLE_MODELS

load_dotenv()

ANALYZER_SCHEMA_KEYS = [
    'audience_profile',
    'expert_persona_name',
    'expert_persona_details',
    'decomposition_depth',
    'delimiter_style',
    'indicator_style',
    'template_style',
    'penalty_text',
    'echo_repeat',
    'few_shot_examples',
]

ANALYZER_FALLBACK_MODELS = [
    'openai/gpt-4o-mini',
    'openai/gpt-4o',
]

def generate_enhanced_prompt(api_key: str, model_id: str, system_prompt: str, original_prompt: str) -> str:
    """
    Dispatches the optimization request to OpenRouter.
    Falls back to environment variables if a personal API key is not supplied.
    """
    # Use user provided key first; fallback to system key if configured
    active_key = api_key.strip() if api_key else os.getenv("OPENROUTER_API_KEY")
    
    if not active_key:
        raise ValueError("Missing API Authorization Key. Provide an OpenRouter key to continue.")

    # Instantiate client pointing to the OpenRouter gateway
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=active_key,
        default_headers={
            "HTTP-Referer": "https://prompt-enhancer.streamlit.app", # Replace with your production domain later
            "X-Title": "Production Prompt Enhancer"
        }
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original Input Prompt:\n{original_prompt}"}
            ],
            temperature=0.3 # Low temperature guarantees adherence to optimization syntax rules
        )

    except Exception as exc:
        # Provide clearer guidance for common failure modes (invalid model, billing)
        msg = str(exc)
        lowered = msg.lower()

        # Invalid model id guidance
        if "not a valid model id" in msg or "invalid model" in lowered:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(
                f"The selected model '{model_id}' is not supported by the OpenRouter gateway. "
                f"Choose a different model from the available options: {available}"
            ) from exc

        # Billing / insufficient credits guidance (HTTP 402)
        if "insufficient credits" in lowered or "402" in lowered:
            raise ValueError(
                "OpenRouter reports insufficient credits for the provided API key. "
                "Please verify the key belongs to the correct account/org and purchase credits at: "
                "https://openrouter.ai/settings/credits"
            ) from exc

        # Generic fallback: re-raise original exception
        raise

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("The model returned an empty response.")

    return content


def _extract_json_from_text(content: str) -> str | None:
    if not content:
        return None

    matches = re.findall(r"\{.*\}", content, flags=re.DOTALL)
    if not matches:
        return None

    return matches[-1].strip()


def _validate_analyzer_schema(parsed: dict, allowed_options: dict | None = None) -> bool:
    if not isinstance(parsed, dict):
        return False

    for key in ANALYZER_SCHEMA_KEYS:
        if key not in parsed:
            return False

    str_keys = [
        'audience_profile',
        'expert_persona_name',
        'expert_persona_details',
        'delimiter_style',
        'indicator_style',
        'template_style',
        'penalty_text',
        'few_shot_examples',
    ]
    for key in str_keys:
        value = parsed.get(key)
        if value is not None and not isinstance(value, str):
            return False

    if not isinstance(parsed.get('decomposition_depth'), int):
        return False
    if not isinstance(parsed.get('echo_repeat'), bool):
        return False

    if allowed_options:
        for key, choices in allowed_options.items():
            value = parsed.get(key)
            if value is not None and value not in choices:
                return False

    return True


def analyze_prompt(api_key: str, base_prompt: str, allowed_options: dict | None = None, model_id: str | list[str] = "openai/gpt-4o-mini") -> str:
    """
    Lightweight analyzer that reads the base prompt and returns a validated JSON string
    containing inferred advanced parameters. `allowed_options` should be a dict
    mapping field names to allowed values.

    This function enforces JSON-only responses and attempts a fallback model if
    the first analyzer fails to produce valid JSON.
    """
    active_key = api_key.strip() if api_key else os.getenv("OPENROUTER_API_KEY")
    if not active_key:
        raise ValueError("Missing API Authorization Key. Provide an OpenRouter key to continue.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=active_key,
    )

    # Build the system prompt enforcing strict JSON output and allowed options
    allowed_options = allowed_options or {}
    opts_text = ""
    for k, v in allowed_options.items():
        opts_text += f"For the key '{k}', choose exactly one of: {v}.\\n"

    schema_keys = [
        'audience_profile','expert_persona_name','expert_persona_details','decomposition_depth',
        'delimiter_style','indicator_style','template_style','penalty_text','echo_repeat','few_shot_examples'
    ]

    system_prompt = (
        "You are a Prompt Engineering Analyzer. Read the user's base draft and infer the best advanced prompt parameters.\n"
        "IMPORTANT: Respond with a single valid JSON object and NOTHING ELSE. The JSON MUST use these exact keys in any order: "
        f"{schema_keys}.\n"
        "Values must be simple types: strings, integers, or booleans. Use null for unknown optional strings.\n"
        "Do not include commentary, explanation, or any text outside the JSON.\n"
        f"{opts_text}"
    )

    candidate_models = [model_id] if isinstance(model_id, str) else list(model_id)
    attempts = 2
    last_content = None
    for candidate_model in candidate_models:
        response = client.chat.completions.create(
            model=candidate_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_prompt}
            ],
            temperature=0.0
        )

        content = response.choices[0].message.content
        last_content = content
        if not content:
            continue

        # Try to extract a JSON object from the assistant reply
        matches = re.findall(r"(\{.*\})", content, flags=re.DOTALL)
        if matches:
            # Prefer the last JSON-like block
            json_str = matches[-1]
            return json_str

        # If no JSON found and we have another attempt, tighten instruction and retry
        system_prompt = (
            "You must respond ONLY with a single valid JSON object that matches the required schema. "
            "Do not include any extra text.\n"
        )

    # If we exit loop without valid JSON, raise with the raw content for debugging
    raise ValueError(f"Analyzer failed to produce JSON. Last response:\n{last_content}")