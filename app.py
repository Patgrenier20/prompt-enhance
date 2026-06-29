# app.py

import streamlit as st
from utils.prompt_manager import AVAILABLE_MODELS, build_meta_prompt
import json
from utils.openrouter_client import generate_enhanced_prompt, analyze_prompt

# Application Global Configuration
st.set_page_config(
    page_title="Custom Prompt Enhancer Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Custom CSS for clean UI boundaries
st.markdown("""
    <style>
    .stTextArea textarea { font-family: monospace; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Multi-Model Prompt Engineering Studio")
st.caption("Re-engineered open-access prompt platform connecting your workflow to top open-source and commercial models.")

# --- SIDEBAR AUTHENTICATION ---
with st.sidebar:
    st.header("🔐 Authentication")
    user_api_key = st.text_input(
        "OpenRouter API Key", 
        type="password", 
        help="Generate an access token via your OpenRouter.ai profile panel."
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.info(
        "This tool evaluates your baseline instructions, applies strict "
        "Prompt Engineering design frameworks, and builds an optimized systemic variant."
    )

# --- ENGINE SETTINGS & STRUCTURAL OPTIONS ---
selected_label = st.selectbox("🤖 Optimization Model Engine", list(AVAILABLE_MODELS.keys()))
target_model_id = AVAILABLE_MODELS[selected_label]

st.subheader("⚙️ Select Optimizing Criteria")
st.markdown(
    "For detailed information on each prompt engineering skill, click "
    "[here](https://arxiv.org/pdf/2312.16171.pdf)"
)

with st.expander("Prompt Engineering Skills"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Prompt Structure & Clarity**")
        skills = {
            "audience": st.checkbox("Audience Integration", help="Incorporates the intended audience directly into the prompt."),
            "affirmative": st.checkbox("Affirmative Sentencing", help="Uses positive, task-oriented phrasing instead of negative framing."),
            "delimiters": st.checkbox("Data Segmentation", help="Separates prompt components using clear delimiters or markdown sections."),
            "output_primer": st.checkbox("Output Primers", help="Includes explicit sample output guidance or examples."),
            "formatted_prompt": st.checkbox("Formatted Prompt", help="Structures the prompt as a clearly formatted instruction template.")
        }

    with col2:
        st.markdown("**Specificity & Instruction**")
        skills.update({
            "few_shot": st.checkbox("Few-Shot Prompting", help="Adds concrete examples that demonstrate the desired response style."),
            "guidance_indicators": st.checkbox("Guidance Indicators", help="Clarifies instructions using keywords, rules, and explicit lists."),
            "no_polite": st.checkbox("No Politeness", help="Removes greetings, pleasantries, and unnecessary polite language."),
            "imperative": st.checkbox("Imperative Task", help="Enforces strong directive phrasing such as 'You MUST' or 'Complete the following'."),
            "penalty_warning": st.checkbox("Penalty Warning", help="Adds a warning for failure to follow instructions or formatting rules.")
        })

    st.markdown("---")
    st.markdown("**Content & Complex Task Support**")
    col3, col4 = st.columns(2)

    with col3:
        skills.update({
            "role_assignment": st.checkbox("Role Assignment", help="Assigns a specific expert persona to the downstream model."),
            "echo_directive": st.checkbox("Echo Directive", help="Repeats key instructions or phrases multiple times for emphasis."),
            "task_decomposition": st.checkbox("Task Decomposition", help="Breaks a complex request into a clear sequence of smaller tasks.")
        })

    with col4:
        st.markdown("\n")
        st.info(
            "Use these options to make the generated prompt more precise, structured, and enforceable. "
            "Combining multiple techniques usually yields the strongest result."
        )

    if skills.get("imperative") and skills.get("penalty_warning"):
        st.warning(
            "Warning: Some prompt engineering skills can conflict with each other and may not be applied effectively. "
            "For example, 'Imperative Task' and 'Penalty Warning' can produce overlapping or redundant enforcement.")

st.markdown("---")

# --- Session state initialization (Phase 1) ---
defaults = {
    "audience_profile": "",
    "expert_persona_name": "",
    "expert_persona_details": "",
    "decomposition_depth": 3,
    "delimiter_style": "Markdown Headings",
    "indicator_style": "Bullet List",
    "template_style": "Markdown Sections",
    "penalty_text": "Failure to follow these instructions will invalidate the output.",
    "echo_repeat": False,
    "few_shot_examples": "",
    "input_prompt": "",
    # auto-filled flags track whether a value was populated by the analyzer
    "auto_audience_profile": False,
    "auto_expert_persona_name": False,
    "auto_expert_persona_details": False,
    "auto_decomposition_depth": False,
    "auto_delimiter_style": False,
    "auto_indicator_style": False,
    "auto_template_style": False,
    "auto_penalty_text": False,
    "auto_echo_repeat": False,
    "auto_few_shot_examples": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# small helper to clear an auto-filled flag when the user edits a field
def _make_clearer(flag_key: str):
    def _clear():
        st.session_state[flag_key] = False
    return _clear

# --- USER RUNTIME LOGIC ---
input_prompt = st.text_area(
    "📝 Base Input Draft",
    key="input_prompt",
    height=160,
    placeholder="Provide the raw text or simple instructions you want optimized..."
)

# Analyzer trigger (Phase 3)
if st.button("Auto-Detect Parameters"):
    if not st.session_state["input_prompt"].strip():
        st.warning("Provide a base input draft first before running auto-detect.")
    else:
        with st.spinner("Analyzing prompt to extract parameters..."):
            # Provide allowed options to the analyzer to prevent hallucinations
            allowed = {
                "delimiter_style": ["Markdown Headings", "XML Tags", "Triple Backticks"],
                "indicator_style": ["Bullet List", "Numbered Steps", "Rule-Based Sections"],
                "template_style": ["Markdown Sections", "Tagged Sections", "Structured Bullets"]
            }
            try:
                raw = analyze_prompt(user_api_key, st.session_state["input_prompt"], allowed_options=allowed)
                parsed = json.loads(raw)
            except Exception as e:
                st.error(f"Analyzer failed: {e}")
            else:
                # Phase 4: validate and update session_state safely
                def _set_with_flag(key, value):
                    st.session_state[key] = value
                    st.session_state[f"auto_{key}"] = True

                if isinstance(parsed, dict):
                    if parsed.get("audience_profile"):
                        _set_with_flag("audience_profile", parsed["audience_profile"])
                    if parsed.get("expert_persona_name"):
                        _set_with_flag("expert_persona_name", parsed["expert_persona_name"])
                    if parsed.get("expert_persona_details"):
                        _set_with_flag("expert_persona_details", parsed["expert_persona_details"])
                    if isinstance(parsed.get("decomposition_depth"), int):
                        _set_with_flag("decomposition_depth", parsed["decomposition_depth"])
                    if parsed.get("delimiter_style") in allowed["delimiter_style"]:
                        _set_with_flag("delimiter_style", parsed["delimiter_style"])
                    if parsed.get("indicator_style") in allowed["indicator_style"]:
                        _set_with_flag("indicator_style", parsed["indicator_style"])
                    if parsed.get("template_style") in allowed["template_style"]:
                        _set_with_flag("template_style", parsed["template_style"])
                    if parsed.get("penalty_text"):
                        _set_with_flag("penalty_text", parsed["penalty_text"])
                    if isinstance(parsed.get("echo_repeat"), bool):
                        _set_with_flag("echo_repeat", parsed["echo_repeat"])
                    if parsed.get("few_shot_examples"):
                        _set_with_flag("few_shot_examples", parsed["few_shot_examples"])
                st.success("Auto-detected parameters populated. Review and adjust before executing.")

with st.expander("Advanced Prompt Parameters", expanded=False):
    if not any(skills.values()):
        st.info("Select one or more prompt engineering skills above to reveal advanced tuning options.")

    if skills.get("audience"):
        label = "Audience Profile"
        if st.session_state.get("auto_audience_profile"):
            label += " (auto-filled)"
        st.text_input(label, key="audience_profile", help="Describe the exact audience or persona the generated prompt should address.", on_change=_make_clearer("auto_audience_profile"))

    if skills.get("output_primer"):
        label = "Example Desired Output"
        if st.session_state.get("auto_few_shot_examples"):
            label += " (auto-filled)"
        st.text_area(label, key="few_shot_examples", height=120, help="Provide a short sample of the desired output style or structure.", on_change=_make_clearer("auto_few_shot_examples"))

    if skills.get("formatted_prompt"):
        label = "Prompt Template Style"
        if st.session_state.get("auto_template_style"):
            label += " (auto-filled)"
        st.selectbox(label, ["Markdown Sections", "Tagged Sections", "Structured Bullets"], key="template_style", on_change=_make_clearer("auto_template_style"))

    if skills.get("role_assignment"):
        label = "Expert Persona Name"
        if st.session_state.get("auto_expert_persona_name"):
            label += " (auto-filled)"
        st.text_input(label, key="expert_persona_name", help="Name the persona to assign in the generated prompt.", on_change=_make_clearer("auto_expert_persona_name"))
        label2 = "Persona Description"
        if st.session_state.get("auto_expert_persona_details"):
            label2 += " (auto-filled)"
        st.text_area(label2, key="expert_persona_details", height=100, help="Add a few detail lines about the persona's expertise, tone, and domain.", on_change=_make_clearer("auto_expert_persona_details"))

    if skills.get("task_decomposition"):
        label = "Decomposition Depth"
        if st.session_state.get("auto_decomposition_depth"):
            label += " (auto-filled)"
        st.slider(label, min_value=2, max_value=6, key="decomposition_depth", on_change=_make_clearer("auto_decomposition_depth"))

    if skills.get("few_shot"):
        label = "Few-Shot Example Prompt"
        if st.session_state.get("auto_few_shot_examples"):
            label += " (auto-filled)"
        st.text_area(label, key="few_shot_examples", height=120, help="Supply one example prompt to guide the generated prompt style.", on_change=_make_clearer("auto_few_shot_examples"))

    if skills.get("guidance_indicators"):
        label = "Guidance Indicator Style"
        if st.session_state.get("auto_indicator_style"):
            label += " (auto-filled)"
        st.selectbox(label, ["Bullet List", "Numbered Steps", "Rule-Based Sections"], key="indicator_style", on_change=_make_clearer("auto_indicator_style"))

    if skills.get("penalty_warning"):
        label = "Penalty Warning Text"
        if st.session_state.get("auto_penalty_text"):
            label += " (auto-filled)"
        st.text_input(label, key="penalty_text", on_change=_make_clearer("auto_penalty_text"))

    if skills.get("echo_directive"):
        label = "Repeat Key Instructions Twice"
        if st.session_state.get("auto_echo_repeat"):
            label += " (auto-filled)"
        st.checkbox(label, key="echo_repeat", on_change=_make_clearer("auto_echo_repeat"))

    if skills.get("delimiters"):
        label = "Delimiter Style"
        if st.session_state.get("auto_delimiter_style"):
            label += " (auto-filled)"
        st.selectbox(label, ["Markdown Headings", "XML Tags", "Triple Backticks"], key="delimiter_style", on_change=_make_clearer("auto_delimiter_style"))

# Action button
action_trigger = st.button("✨ Execute Transformation Pipeline", type="primary", use_container_width=True)

if action_trigger:
    if not input_prompt.strip():
        st.warning("Execution halted: The input draft area cannot be empty.")
        st.stop()

    # Build prompt configurations dynamically from session state
    extra_params = {
        "audience_profile": st.session_state.get("audience_profile"),
        "few_shot_examples": st.session_state.get("few_shot_examples"),
        "template_style": st.session_state.get("template_style"),
        "persona_name": st.session_state.get("expert_persona_name"),
        "persona_details": st.session_state.get("expert_persona_details"),
        "decomposition_depth": st.session_state.get("decomposition_depth"),
        "indicator_style": st.session_state.get("indicator_style"),
        "penalty_text": st.session_state.get("penalty_text"),
        "echo_repeat": st.session_state.get("echo_repeat"),
        "delimiter_style": st.session_state.get("delimiter_style"),
    }

    meta_system_prompt = build_meta_prompt(skills, extra_params)

    with st.spinner(f"Routing payloads to engine: {selected_label}..."):
        try:
            output_prompt = generate_enhanced_prompt(
                api_key=user_api_key,
                model_id=target_model_id,
                system_prompt=meta_system_prompt,
                original_prompt=input_prompt
            )
            
            st.success("🎉 Prompt Optimization Complete!")
            
            # Display transformed text asset in clean, interactive layout
            st.text_area(
                label="✨ Production Optimized Output",
                value=output_prompt,
                height=300,
                key="optimized_output"
            )
            st.caption("💡 Copy the generated text blocks above directly into your production pipelines.")

        except Exception as error_msg:
            st.error(f"Execution Error Intercepted: {error_msg}")