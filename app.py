"""ATS Resume Checker - Streamlit UI + Gemini Flash.

Upload a resume (PDF or DOCX) and get:
  * an overall ATS score (0-100) with a category breakdown
  * strengths, prioritised improvements and missing keywords
  * optional tailoring to a pasted job description
"""

import io
import json
import os
import re

import streamlit as st

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.8-flash"  # override with GEMINI_MODEL (env or secret)
MAX_FILE_MB = 5
MAX_CHARS = 15000  # resume text sent to the model
MIN_CHARS = 150  # below this we assume the PDF is scanned / empty

# Category weights (must add up to 100). The overall score is computed here
# in code so it is consistent and not left entirely to the model.
WEIGHTS = {
    "keywords_relevance": 30,
    "content_impact": 25,
    "formatting_parsability": 20,
    "structure_sections": 15,
    "readability_length": 10,
}
LABELS = {
    "keywords_relevance": "Keywords & relevance",
    "content_impact": "Content & impact",
    "formatting_parsability": "Formatting & parsability",
    "structure_sections": "Structure & sections",
    "readability_length": "Readability & length",
}


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
def extract_text(file_bytes: bytes, filename: str) -> str:
    """Return plain text from a PDF or DOCX file. Raises ValueError on bad input."""
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise ValueError("This PDF is password-protected.")
            pages = [(page.extract_text() or "") for page in reader.pages]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read this PDF: {exc}")
        return "\n".join(pages).strip()

    if name.endswith(".docx"):
        from docx import Document

        try:
            doc = Document(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError(f"Could not read this DOCX: {exc}")
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        # Many resumes keep content inside tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text.strip())
        return "\n".join(parts).strip()

    raise ValueError("Unsupported file type. Please upload a PDF or DOCX.")


# --------------------------------------------------------------------------
# Prompt + response handling
# --------------------------------------------------------------------------
def build_prompt(resume_text: str, job_description: str = "") -> str:
    jd_block = (
        f"\nJOB DESCRIPTION (score keyword match against this):\n\"\"\"\n{job_description.strip()[:6000]}\n\"\"\"\n"
        if job_description and job_description.strip()
        else "\nNo job description was provided. Judge keywords against general "
        "industry standards for the role the resume appears to target.\n"
    )
    return f"""You are an expert ATS (Applicant Tracking System) analyst and professional resume reviewer.
Evaluate the resume below the way a strict ATS parser plus a recruiter would.

Score each category from 0 to 100 (integers):
- keywords_relevance: relevant hard/soft skills, tools, job-title keywords{" matched to the job description" if job_description and job_description.strip() else ""}
- content_impact: quantified achievements, action verbs, results rather than duties
- formatting_parsability: clean layout, standard headings, no signs of tables/columns/graphics that break ATS parsing, consistent dates, contact details present
- structure_sections: presence and order of Contact, Summary, Experience, Education, Skills, Projects/Certifications
- readability_length: concise bullets, no typos or grammar problems, sensible length

Be honest and critical. Do not inflate scores. Only reference things that are actually in the resume.
{jd_block}
RESUME TEXT:
\"\"\"
{resume_text[:MAX_CHARS]}
\"\"\"

Respond with ONLY a JSON object, no markdown, in exactly this shape:
{{
  "detected_role": "short string",
  "category_scores": {{
    "keywords_relevance": 0,
    "content_impact": 0,
    "formatting_parsability": 0,
    "structure_sections": 0,
    "readability_length": 0
  }},
  "summary": "2-3 sentence overall assessment",
  "strengths": ["..."],
  "improvements": [
    {{"priority": "High|Medium|Low", "issue": "what is wrong", "fix": "specific, actionable fix", "example": "optional rewritten bullet or line, else empty string"}}
  ],
  "missing_keywords": ["..."],
  "ats_formatting_warnings": ["..."]
}}
Give 3-6 strengths, 5-10 improvements ordered by priority, and up to 15 missing keywords."""


def parse_json_response(text: str) -> dict:
    """Parse model output into a dict, tolerating code fences or stray text."""
    if not text:
        raise ValueError("The model returned an empty response.")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The model response was not valid JSON.")
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            raise ValueError("The model response was not valid JSON.")
    if not isinstance(data, dict):
        raise ValueError("Unexpected response format from the model.")
    return data


def _clamp(value, default=0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _str_list(value, limit=20) -> list:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


def normalize_result(data: dict) -> dict:
    """Validate/clean model output and compute the weighted overall score."""
    raw_scores = data.get("category_scores") or {}
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    scores = {key: _clamp(raw_scores.get(key)) for key in WEIGHTS}
    overall = round(sum(scores[k] * w for k, w in WEIGHTS.items()) / sum(WEIGHTS.values()))

    improvements = []
    order = {"high": 0, "medium": 1, "low": 2}
    for item in data.get("improvements") or []:
        if not isinstance(item, dict):
            continue
        priority = str(item.get("priority", "Medium")).strip().capitalize()
        if priority.lower() not in order:
            priority = "Medium"
        issue = str(item.get("issue", "")).strip()
        fix = str(item.get("fix", "")).strip()
        if not (issue or fix):
            continue
        improvements.append(
            {
                "priority": priority,
                "issue": issue,
                "fix": fix,
                "example": str(item.get("example", "") or "").strip(),
            }
        )
    improvements.sort(key=lambda i: order[i["priority"].lower()])

    return {
        "overall": overall,
        "category_scores": scores,
        "detected_role": str(data.get("detected_role", "") or "").strip(),
        "summary": str(data.get("summary", "") or "").strip(),
        "strengths": _str_list(data.get("strengths")),
        "improvements": improvements,
        "missing_keywords": _str_list(data.get("missing_keywords"), 15),
        "warnings": _str_list(data.get("ats_formatting_warnings")),
    }


def analyze_resume(resume_text: str, job_description: str, api_key: str, model: str) -> dict:
    """Call Gemini and return the normalized analysis."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(resume_text, job_description),
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    return normalize_result(parse_json_response(response.text))


# --------------------------------------------------------------------------
# Helpers for UI
# --------------------------------------------------------------------------
def score_band(score: int):
    if score >= 80:
        return "Excellent", "🟢"
    if score >= 65:
        return "Good - room to improve", "🟡"
    if score >= 50:
        return "Needs work", "🟠"
    return "Poor - major fixes needed", "🔴"


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets or environment without crashing if neither exists."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")
    st.title("📄 ATS Resume Checker")
    st.caption("Upload your resume to get an ATS score and specific improvements, powered by Gemini.")

    with st.sidebar:
        st.header("Settings")
        api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        if api_key:
            st.success("API key loaded from secrets.")
        else:
            api_key = st.text_input(
                "Gemini API key",
                type="password",
                help="Get a free key at https://aistudio.google.com/apikey",
            )
        model = st.text_input("Model", value=get_secret("GEMINI_MODEL", DEFAULT_MODEL))
        st.markdown("---")
        st.caption("Your resume is sent to the Gemini API for analysis and is not stored by this app.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
    with col_right:
        job_description = st.text_area(
            "Job description (optional)",
            height=150,
            placeholder="Paste a job description to check how well your resume matches it...",
        )

    if st.button("Analyze resume", type="primary", disabled=uploaded is None):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            st.stop()
        if not model.strip():
            st.error("Please enter a model name in the sidebar.")
            st.stop()

        file_bytes = uploaded.getvalue()
        if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_FILE_MB} MB. Please upload a smaller file.")
            st.stop()

        result, error = None, None
        try:
            with st.spinner("Reading your resume..."):
                text = extract_text(file_bytes, uploaded.name)
            if len(text) < MIN_CHARS:
                raise ValueError(
                    "Very little text could be extracted. If this is a scanned/image PDF, "
                    "an ATS can't read it either - export a text-based PDF or DOCX and retry."
                )
            with st.spinner("Analyzing with Gemini..."):
                result = analyze_resume(text, job_description, api_key, model.strip())
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:  # network, quota, invalid key, bad model name...
            error = f"Analysis failed: {exc}"

        if error:
            st.error(error)
            st.stop()

        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        st.info("Upload a resume and click **Analyze resume** to begin.")
        return

    overall = result["overall"]
    band, icon = score_band(overall)
    st.divider()
    top_l, top_r = st.columns([1, 2])
    with top_l:
        st.metric("ATS score", f"{overall} / 100")
        st.progress(overall / 100)
        st.markdown(f"{icon} **{band}**")
        if result["detected_role"]:
            st.caption(f"Detected target role: {result['detected_role']}")
    with top_r:
        st.subheader("Summary")
        st.write(result["summary"] or "No summary returned.")

    st.subheader("Score breakdown")
    cols = st.columns(len(WEIGHTS))
    for col, key in zip(cols, WEIGHTS):
        with col:
            st.metric(LABELS[key], result["category_scores"][key])
            st.progress(result["category_scores"][key] / 100)
            st.caption(f"Weight: {WEIGHTS[key]}%")

    tab_fix, tab_good, tab_kw = st.tabs(["🔧 Improvements", "✅ Strengths", "🔑 Keywords & warnings"])

    with tab_fix:
        if not result["improvements"]:
            st.write("No improvements returned.")
        for item in result["improvements"]:
            marker = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}[item["priority"]]
            with st.expander(f"{marker} {item['priority']} - {item['issue'] or 'Improvement'}"):
                st.markdown(f"**Fix:** {item['fix']}")
                if item["example"]:
                    st.markdown("**Example:**")
                    st.code(item["example"], language=None)

    with tab_good:
        if result["strengths"]:
            for s in result["strengths"]:
                st.markdown(f"- {s}")
        else:
            st.write("No strengths returned.")

    with tab_kw:
        st.markdown("**Missing keywords to consider adding (only where truthful):**")
        if result["missing_keywords"]:
            st.write(", ".join(f"`{k}`" for k in result["missing_keywords"]))
        else:
            st.write("None identified.")
        st.markdown("**ATS formatting warnings:**")
        if result["warnings"]:
            for w in result["warnings"]:
                st.markdown(f"- {w}")
        else:
            st.write("None identified.")

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(result, indent=2),
        file_name="ats_report.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
