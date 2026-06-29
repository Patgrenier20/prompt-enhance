# utils/prompt_manager.py

AVAILABLE_MODELS = {
    "GPT-4o (OpenAI)": "openai/gpt-4o",
    "GPT-4o Mini (OpenAI)": "openai/gpt-4o-mini",
    "Claude 3.5 Sonnet (Anthropic)": "anthropic/claude-3.5-sonnet",
    "Claude 3 Haiku (Anthropic)": "anthropic/claude-3-haiku",
    "Gemini 1.5 Pro (Google)": "google/gemini-1.5-pro",
    "Gemini 1.5 Flash (Google)": "google/gemini-1.5-flash",
    "Llama 3 70B (Meta)": "meta-llama/llama-3-70b-instruct",
    "Mistral Large (Mistral)": "mistralai/mistral-large"
}

def build_meta_prompt(skills_dict: dict, extra_params: dict | None = None, *args, **kwargs) -> str:
    """
    Compiles selected prompt engineering techniques and advanced parameter values
    into a unified instruction set for the optimizing model.
    """
    if extra_params is None and args:
        extra_params = args[0]
    if extra_params is None:
        extra_params = {}
    # 1. STOP PARAPHRASING: Give it a strict persona and tell it exactly what NOT to do.
    system_instructions = (
        "You are an elite, master-level Prompt Engineer. The user will provide a basic text draft or idea.\n"
        "CRITICAL INSTRUCTION: You are NOT allowed to simply rewrite, paraphrase, or beautify the user's original text. "
        "The user's text is raw input only; your job is to construct a completely new, professional prompt for a downstream model.\n\n"

        "Your output MUST be a finished prompt written as a standalone instruction set. It must include these sections, using Markdown headings exactly as shown:\n"
        "### 🎭 System Role & Context\n"
        "- Define the persona, expertise, and operational context for the downstream model.\n\n"
        "### 🎯 Primary Directive\n"
        "- State the exact task the downstream model must perform in clear direct language.\n\n"
        "### 🚧 Strict Constraints & Guidelines\n"
        "- List explicit rules, restrictions, formatting requirements, and quality expectations as bullet points.\n\n"
        "### 📤 Desired Output Format\n"
        "- Describe the exact structure, tone, and deliverables the downstream model should produce.\n\n"

        "Do NOT include any analysis, commentary, or explanation in the final output. Return only the finalized prompt that a different LLM can use directly.\n"
        "Do NOT reuse more than two words from the user's original input in sequence unless those words are exact domain terms.\n\n"

        "Apply the following prompt engineering techniques based on the user's selected options:\n"
    )
    
    # Structural Mapping
    if skills_dict.get("audience"):
        system_instructions += "- Add a constraint specifying the exact target audience and their technical literacy level.\n"
    if skills_dict.get("affirmative"):
        system_instructions += "- Write all constraints using affirmative, task-oriented language.\n"
    if skills_dict.get("delimiters"):
        system_instructions += "- Instruct the AI to separate prompt sections using clear delimiters, headings, or markdown blocks.\n"
    if skills_dict.get("output_primer"):
        system_instructions += "- Include explicit output examples or sample response structure within the generated prompt.\n"
    if skills_dict.get("formatted_prompt"):
        system_instructions += "- Build the final prompt as a clearly formatted template with distinct sections and labels.\n"

    # Behavioral Mapping
    if skills_dict.get("role_assignment"):
        system_instructions += "- Populate the 'System Role' section with a highly detailed, world-class expert persona.\n"
    if skills_dict.get("imperative"):
        system_instructions += "- Use direct imperative phrasing such as 'You MUST' and 'Complete the following task'.\n"
    if skills_dict.get("no_polite"):
        system_instructions += "- Add a strict constraint forbidding the AI from using conversational filler, greetings, or apologies.\n"
    if skills_dict.get("penalty_warning"):
        system_instructions += "- Include a warning that failure to follow the rules will result in incorrect output.\n"
    if skills_dict.get("echo_directive"):
        system_instructions += "- Repeat the most important instructions or phrases at least twice to ensure emphasis.\n"
    if skills_dict.get("few_shot"):
        system_instructions += "- Provide one or two short examples of the desired response style or output structure.\n"
    if skills_dict.get("guidance_indicators"):
        system_instructions += "- Use explicit indicators like bullet lists, numbered steps, and rule-based keywords in the prompt.\n"
    if skills_dict.get("task_decomposition"):
        system_instructions += "- Break the user's request into a sequence of smaller, easier-to-follow sub-tasks where appropriate.\n"

    # Context Modifiers
    if skills_dict.get("step_by_step"):
        system_instructions += "- Add a constraint forcing the AI to explicitly use 'Chain of Thought' reasoning before outputting the final answer.\n"
    if skills_dict.get("beginner"):
        system_instructions += "- Add a constraint forcing the model to explain complex terminology as if addressing a complete beginner.\n"
    if skills_dict.get("unbiased"):
        system_instructions += "- Enforce strict neutrality in the constraints, commanding the elimination of bias or assumption.\n"

    # Advanced parameter values
    if extra_params.get("audience_profile"):
        system_instructions += f"- Specify the target audience: {extra_params['audience_profile']}.\n"
    if extra_params.get("output_example"):
        system_instructions += "- Include an example of the desired output format or structure in the generated prompt.\n"
    if extra_params.get("template_style"):
        system_instructions += f"- Use the following prompt template style: {extra_params['template_style']}.\n"
    if extra_params.get("persona_name") or extra_params.get("persona_details"):
        persona = extra_params.get("persona_name", "Expert persona")
        details = extra_params.get("persona_details", "A highly competent specialist.")
        system_instructions += f"- Set the 'System Role' to a persona named {persona} with these details: {details}\n"
    if extra_params.get("decomposition_depth"):
        system_instructions += f"- Decompose the user's request into {extra_params['decomposition_depth']} smaller, ordered subtasks.\n"
    if extra_params.get("few_shot_examples"):
        system_instructions += "- Incorporate the provided few-shot example prompt to guide the style and structure.\n"
    if extra_params.get("indicator_style"):
        system_instructions += f"- Use {extra_params['indicator_style']} to make the instructions and constraints easier to follow.\n"
    if extra_params.get("penalty_text"):
        system_instructions += f"- Add a penalty rule: {extra_params['penalty_text']}\n"
    if extra_params.get("echo_repeat"):
        system_instructions += "- Repeat the most important instructions or rules twice to reinforce them in the prompt.\n"
    if extra_params.get("delimiter_style"):
        system_instructions += f"- Use {extra_params['delimiter_style']} to clearly separate sections of the generated prompt.\n"

    # 2. STRICT OUTPUT ENFORCEMENT
    system_instructions += (
        "\nFINAL RULES:\n"
        "1. Return ONLY the finalized, ready-to-copy engineered prompt.\n"
        "2. Do not include any greeting, introduction, or meta-commentary like 'Here is your enhanced prompt'.\n"
        "3. Return a complete standalone prompt that a different LLM can use directly.\n"
        "4. If the input is a research or market analysis request, include explicit instructions for structure, examples, and the expected result format.\n"
    )
    
    return system_instructions